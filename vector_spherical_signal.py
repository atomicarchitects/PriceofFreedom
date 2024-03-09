from typing import Callable
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import _s2grid, _s2grid_vectors, parity_function


class VectorSphericalSignal:

    def __init__(
        self,
        grid_values: e3nn.IrrepsArray,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        _perform_checks: bool = True,
    ) -> None:
        if _perform_checks:
            if len(grid_values.shape) < 2:
                raise ValueError(
                    f"Grid values should have atleast 2 axes. Got grid_values of shape {grid_values.shape}."
                )

            if quadrature not in ["soft", "gausslegendre"]:
                raise ValueError(
                    f"Invalid quadrature for SphericalSignal: {quadrature}"
                )

            if p_val not in (-1, 1):
                raise ValueError(
                    f"Parity p_val must be either +1 or -1. Received: {p_val}"
                )

            if p_arg not in (-1, 1):
                raise ValueError(
                    f"Parity p_arg must be either +1 or -1. Received: {p_arg}"
                )

        self.grid_values = grid_values
        self.quadrature = quadrature
        self.p_val = p_val
        self.p_arg = p_arg

    @staticmethod
    def zeros(
        res_beta: int,
        res_alpha: int,
        quadrature: str,
        lmax: int,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        dtype: jnp.dtype = jnp.float32,
    ) -> "VectorSphericalSignal":
        """Create a null signal on a grid."""
        grid_values = e3nn.zeros(
            e3nn.s2_irreps(lmax, p_val=p_val, p_arg=p_arg),
            leading_shape=(res_beta, res_alpha),
            dtype=dtype,
        )
        return VectorSphericalSignal(
            grid_values,
            quadrature,
            p_val=p_val,
            p_arg=p_arg,
        )

    @staticmethod
    def from_function(
        func: Callable[[jax.Array], e3nn.IrrepsArray],
        res_beta: int,
        res_alpha: int,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
    ) -> "VectorSphericalSignal":
        """Create a signal on the sphere from a function of the coordinates.

        Args:
            func (`Callable`): function on the sphere that maps a 3-dimensional array (x, y, z) to a IrrepsArray
            res_beta: resolution for beta
            res_alpha: resolution for alpha
            quadrature: quadrature to use
            p_val: parity of the signal, either +1 or -1
            p_arg: parity of the argument of the signal, either +1 or -1
            dtype: dtype of the signal

        Returns:
            `VectorSphericalSignal`: signal on the sphere
        """
        y, alpha, _ = _s2grid(res_beta, res_alpha, quadrature)
        grid_vectors = _s2grid_vectors(y, alpha)
        grid_values = jax.vmap(jax.vmap(func))(grid_vectors)
        grid_values = e3nn.as_irreps_array(grid_values)
        return VectorSphericalSignal(grid_values, quadrature, p_val=p_val, p_arg=p_arg)

    @property
    def shape(self) -> tuple:
        """Shape of the grid."""
        return self.grid_values.shape

    @property
    def res_beta(self) -> int:
        """Grid resolution for beta."""
        return self.grid_values.shape[-3]

    @property
    def res_alpha(self) -> int:
        """Grid resolution for alpha."""
        return self.grid_values.shape[-2]

    @property
    def grid_vectors(self) -> jax.Array:
        """Returns the coordinates of the points on the sphere. Shape: ``(res_beta, res_alpha, 3)``."""
        y, alpha, _ = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return _s2grid_vectors(y, alpha)

    def dot(self, other: "VectorSphericalSignal") -> e3nn.IrrepsArray:
        """Dot product of two signals."""
        if self.shape != other.shape:
            raise ValueError(
                f"Shapes of the two signals do not match: {self.shape} and {other.shape}"
            )
        return e3nn.SphericalSignal(
            e3nn.dot(self.grid_values, other.grid_values),
            quadrature=self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def cross(self, other: "VectorSphericalSignal") -> "VectorSphericalSignal":
        """Cross product of two signals."""
        if self.shape != other.shape:
            raise ValueError(
                f"Shapes of the two signals do not match: {self.shape} and {other.shape}"
            )
        return self.replace_values(e3nn.cross(self.grid_values, other.grid_values))

    def apply(
        self, func: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray]
    ) -> "VectorSphericalSignal":
        """Applies a function pointwise on the grid."""
        new_p_val = parity_function(func) if self.p_val == -1 else self.p_val
        if new_p_val == 0:
            raise ValueError(
                "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
            )
        return self.replace_values(grid_values=func(self.grid_values))

    def replace_values(self, grid_values: e3nn.IrrepsArray) -> "VectorSphericalSignal":
        """Replace the grid values of the signal."""
        return VectorSphericalSignal(
            grid_values, self.quadrature, p_val=self.p_val, p_arg=self.p_arg
        )