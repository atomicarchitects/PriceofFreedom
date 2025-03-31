"""Definition of a simple E(3)-equivariant model."""

from typing import Callable, Tuple, Sequence
import flax.linen as nn

import jax
import jax.numpy as jnp
import jraph
import e3nn_jax as e3nn

from src.models import mlp


def compute_features_of_relative_vectors(
    relative_vectors: jnp.ndarray, lmax: int
) -> Tuple[e3nn.IrrepsArray, jnp.ndarray]:
    """Compute the spherical harmonics of the relative vectors and their norms."""
    relative_vectors_sh = e3nn.spherical_harmonics(
        e3nn.s2_irreps(lmax=lmax),
        relative_vectors,
        normalize=True,
        normalization="norm",
    )
    relative_vectors_norm = jnp.linalg.norm(relative_vectors, axis=-1, keepdims=True)
    return relative_vectors_sh, relative_vectors_norm


class SimpleNetworkLayer(nn.Module):
    """A layer of a simple E(3)-equivariant message passing network."""

    mlp_hidden_dims: int
    mlp_num_layers: int
    output_irreps: e3nn.Irreps
    tensor_product_fn: Callable[[], nn.Module]

    @nn.compact
    def __call__(
        self,
        node_features: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        relative_vectors_sh: e3nn.IrrepsArray,
        relative_vectors_norm: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        # Compute the skip connection.
        node_features_skip = node_features

        # Tensor product of the relative vectors and the neighbouring node features.
        node_features_broadcasted = node_features[senders]
        node_features_broadcasted = self.tensor_product_fn()(relative_vectors_sh, node_features_broadcasted)

        # Simply multiply each irrep by a learned scalar, based on the norm of the relative vector.
        scalars = mlp.MLP(
            output_dims=node_features_broadcasted.irreps.num_irreps,
            hidden_dims=self.mlp_hidden_dims,
            num_layers=self.mlp_num_layers,
        )(relative_vectors_norm)
        scalars = e3nn.IrrepsArray(f"{scalars.shape[-1]}x0e", scalars)
        node_features_broadcasted = jax.vmap(lambda scale, feature: scale * feature)(scalars, node_features_broadcasted)

        # Aggregate the node features back.
        node_features = e3nn.scatter_mean(
            node_features_broadcasted,
            dst=receivers,
            output_size=node_features.shape[0],
        )

        # Apply a non-linearity.
        # Note that using an unnormalized non-linearity will make the model not equivariant.
        gate_irreps = e3nn.Irreps(f"{node_features.irreps.num_irreps}x0e")
        node_features_expanded = e3nn.flax.Linear(node_features.irreps + gate_irreps)(node_features)
        node_features = e3nn.gate(node_features_expanded)

        # Add the skip connection.
        node_features = e3nn.concatenate([node_features, node_features_skip])

        # Apply a linear transformation to the output.
        node_features = e3nn.flax.Linear(self.output_irreps)(node_features)
        return node_features


class AtomEmbedding(nn.Module):
    """Embeds atomic numbers into a learnable vector space."""

    embed_dims: int
    max_atomic_number: int

    @nn.compact
    def __call__(self, atomic_numbers: jnp.ndarray) -> jnp.ndarray:
        atom_embeddings = nn.Embed(num_embeddings=self.max_atomic_number, features=self.embed_dims)(atomic_numbers)
        return e3nn.IrrepsArray(f"{self.embed_dims}x0e", atom_embeddings)


class SimpleNetwork(nn.Module):
    """A simple E(3)-equivariant message passing network."""

    sh_lmax: int
    init_embed_dims: int
    max_atomic_number: int
    mlp_hidden_dims: int
    mlp_num_layers: int
    output_irreps_per_layer: Sequence[e3nn.Irreps]
    tensor_product_fn: Callable[[], nn.Module]
    readout: nn.Module

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jnp.ndarray:
        # Node features are initially the atomic numbers embedded.
        node_features = graphs.nodes["numbers"]
        node_features = AtomEmbedding(
            embed_dims=self.init_embed_dims,
            max_atomic_number=self.max_atomic_number,
        )(node_features)

        # Precompute the spherical harmonics of the relative vectors.
        positions = graphs.nodes["positions"]
        relative_vectors = positions[graphs.receivers] - positions[graphs.senders]
        relative_vectors_sh, relative_vectors_norm = compute_features_of_relative_vectors(
            relative_vectors,
            lmax=self.sh_lmax,
        )

        # Message passing.
        for output_irreps in self.output_irreps_per_layer:
            node_features = SimpleNetworkLayer(
                mlp_hidden_dims=self.mlp_hidden_dims,
                mlp_num_layers=self.mlp_num_layers,
                tensor_product_fn=self.tensor_product_fn,
                output_irreps=output_irreps,
            )(
                node_features,
                graphs.senders,
                graphs.receivers,
                relative_vectors_sh,
                relative_vectors_norm,
            )

        # Readout.
        return self.readout(node_features, graphs)
