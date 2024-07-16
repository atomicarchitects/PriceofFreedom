import flax.linen as nn
import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph

from src.models import mlp

class TetrisReadout(nn.Module):
    """A learnable readout for Tetris."""

    @nn.compact
    def __call__(self, node_features: e3nn.IrrepsArray, graphs: jraph.GraphsTuple) -> jnp.ndarray:
        # Apply a linear layer to the node features.
        node_features = e3nn.flax.Linear("0o + 7x0e", force_irreps_out=True)(node_features)
    
        # Readout logits.
        pred = e3nn.scatter_sum(
            node_features, nel=graphs.n_node
        ).array  # [num_graphs, 1 + 7]
        odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
        logits = jnp.concatenate([odd * even1, -odd * even1, even2], axis=1)
        assert logits.shape == (
            len(graphs.n_node),
            8,
        ), logits.shape  # [num_graphs, num_classes]

        return logits
