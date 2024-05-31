"""Definition of a simple E(3)-equivariant model."""

from typing import Callable
import flax.linen as nn

import jax
import jax.numpy as jnp
import jraph
import e3nn_jax as e3nn


class MLP(nn.Module):

    output_dims: int
    hidden_dims: int = 32
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dims)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        x = nn.Dense(features=self.output_dims)(x)
        return x


class SimpleNetwork(nn.Module):

    sh_lmax: int
    lmax: int
    init_node_features: int
    max_atomic_number: int
    num_hops: int
    output_dims: int
    tensor_product_fn: Callable[[], nn.Module]

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jnp.ndarray:
        # Node features are initially the atomic numbers embedded.
        node_features = graphs.nodes["numbers"]
        node_features = nn.Embed(
            num_embeddings=self.max_atomic_number, features=self.init_node_features
        )(node_features)
        node_features = e3nn.IrrepsArray(f"{self.init_node_features}x0e", node_features)

        # Precompute the spherical harmonics of the relative vectors.
        relative_vectors = graphs.edges["relative_vectors"]
        relative_vectors_sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(lmax=self.sh_lmax),
            relative_vectors,
            normalize=True,
            normalization="norm",
        )
        relative_vectors_norm = jnp.linalg.norm(
            relative_vectors, axis=-1, keepdims=True
        )

        # print("relative_vectors_sh", e3nn.norm(relative_vectors_sh))
        # print("node_features", e3nn.norm(node_features))

        for _ in range(self.num_hops):
            # Tensor product of the relative vectors and the neighbouring node features.
            node_features_broadcasted = node_features[graphs.senders]
            tp = self.tensor_product_fn()(
                relative_vectors_sh, node_features_broadcasted
            )
            tp = tp.filter(lmax=self.lmax)

            # Apply a linear transformation to the tensor product.
            tp = e3nn.flax.Linear(tp.irreps)(tp)

            # Simply multiply each irrep by a learned scalar.
            scalars = MLP(output_dims=tp.irreps.num_irreps)(relative_vectors_norm)
            scalars = e3nn.IrrepsArray(f"{scalars.shape[-1]}x0e", scalars)
            node_features_broadcasted = jax.vmap(lambda sc, feat: sc * feat)(
                scalars, tp
            )

            # Aggregate the node features back.
            node_features = e3nn.scatter_mean(
                node_features_broadcasted,
                dst=graphs.receivers,
                output_size=node_features.shape[0],
            )

        # Global readout.
        graph_globals = e3nn.scatter_mean(node_features.filter("0e"), nel=graphs.n_node)
        return MLP(output_dims=self.output_dims)(graph_globals.array)
