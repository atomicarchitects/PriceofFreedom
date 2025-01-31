import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """A simple multi-layer perceptron."""

    output_dims: int
    hidden_dims: int
    num_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dims)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        x = nn.Dense(features=self.output_dims)(x)
        return x
