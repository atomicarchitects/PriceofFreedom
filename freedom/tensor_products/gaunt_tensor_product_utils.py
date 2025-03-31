from typing import Callable
import functools
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np


class RectangularSignal:
    """A signal defined on a rectangular region as a function of theta and phi."""

    def __init__(self, grid_values: jax.Array, res_theta: int, res_phi: int):
        self.grid_values = grid_values
        if res_theta <= 2 or res_phi <= 2:
            raise ValueError("res_theta and res_phi must be greater than 2.")

        self.res_theta = res_theta
        self.res_phi = res_phi
        assert self.grid_values.shape == (
            *self.grid_values.shape[:-2],
            res_theta,
            res_phi,
        )

    def from_function(
        f: Callable[[jax.Array, jax.Array], jax.Array],
        *,
        res_theta: int,
        res_phi: int,
        wrap_theta: bool,
    ):
        """Create a signal from a function of theta and phi."""
        if wrap_theta:

            def f_wrapped(theta: jax.Array, phi: jax.Array) -> jax.Array:
                f_val = f(theta, phi)
                fd_val = -f(theta, phi)
                return jnp.where(theta < jnp.pi, f_val, fd_val)

            func = f_wrapped
        else:
            func = f
        thetas, phis = jnp.meshgrid(
            RectangularSignal._thetas(res_theta),
            RectangularSignal._phis(res_phi),
            indexing="ij",
        )
        fn_vmap = jax.jit(jax.vmap(jax.vmap(func)))
        grid_values = fn_vmap(thetas, phis)

        return RectangularSignal(grid_values, res_theta, res_phi)

    @staticmethod
    def _thetas(res_theta: int):
        """Returns the theta values of the grid."""
        return jnp.linspace(0, 2 * jnp.pi, res_theta)

    @staticmethod
    def _phis(res_phi: int):
        """Returns the phi values of the grid."""
        return jnp.linspace(0, 2 * jnp.pi, res_phi)

    def thetas(self):
        """Returns the theta values of the grid."""
        return RectangularSignal._thetas(self.res_theta)

    def phis(self):
        """Returns the phi values of the grid."""
        return RectangularSignal._phis(self.res_phi)

    def integrate(self, area_element: str) -> jax.Array:
        """Computes the integral of the signal over a rectangular/spherical region."""
        if area_element == "rectangular":
            return self.integrate_rectangular()
        elif area_element == "spherical":
            return self.integrate_spherical()
        else:
            raise ValueError(f"Unknown area element {area_element}")

    def integrate_rectangular(self) -> jax.Array:
        """Computes the integral of the signal over the rectangular region."""
        return RectangularSignal._integrate(self.grid_values, self.thetas(), self.phis())

    def integrate_spherical(self) -> jax.Array:
        """Computes the integral of the signal over the spherical region."""
        # Only integrate upto theta = pi.
        thetas = self.thetas()[: self.res_theta // 2]
        grid_values = self.grid_values[..., : self.res_theta // 2, :]
        return RectangularSignal._integrate(grid_values * jnp.sin(thetas)[:, None], thetas, self.phis())

    @staticmethod
    def _integrate(grid_values: jax.Array, thetas: jax.Array, phis: jax.Array) -> jax.Array:
        """Computes the integral of the signal over the rectangular region."""
        assert grid_values.shape == (len(thetas), len(phis))

        # Integrate over theta axis first, with the trapezoidal rule.
        # Ideally we would use the symmetry around theta = pi to reduce the number of points to integrate over,
        # but I don't think it's worth the effort for now.
        dtheta = thetas[1] - thetas[0]
        theta_weights = jnp.concatenate([jnp.array([0.5]), jnp.ones(len(thetas) - 2), jnp.array([0.5])])
        integral = jnp.sum(grid_values * theta_weights[:, None], axis=0) * dtheta
        assert integral.shape == (len(phis),)

        # Integrate over phi axis next, with the trapezoidal rule.
        dphi = phis[1] - phis[0]
        phi_weights = jnp.concatenate([jnp.array([0.5]), jnp.ones(len(phis) - 2), jnp.array([0.5])])
        integral = jnp.sum(integral * phi_weights, axis=0) * dphi
        assert integral.shape == ()
        return integral

    def __mul__(self, other):
        """Pointwise multiplication of two signals."""
        assert isinstance(other, RectangularSignal)
        assert self.res_theta == other.res_theta
        assert self.res_phi == other.res_phi
        return RectangularSignal(self.grid_values * other.grid_values, self.res_theta, self.res_phi)


def from_lm_index(lm_index: int) -> tuple:
    """Converts a grid index to l, m values."""
    l = jnp.floor(jnp.sqrt(lm_index)).astype(jnp.int32)
    m = lm_index - l * (l + 1)
    return l, m


def to_lm_index(l: int, m: int) -> int:
    """Converts l, m values to a grid index."""
    return l * (l + 1) + m


def sh_phi(l: int, m: int, phi: float) -> float:
    r"""Phi dependence of spherical harmonics.

    Args:
        l: l value
        phi: phi value

    Returns:
        Array of shape ``(2 * l + 1,)``
    """
    assert phi.ndim == 0
    phi = phi[..., None]  # [..., 1]
    ms = jnp.arange(1, l + 1)  # [1, 2, 3, ..., l]
    cos = jnp.cos(ms * phi)  # [..., m]

    ms = jnp.arange(l, 0, -1)  # [l, l-1, l-2, ..., 1]
    sin = jnp.sin(ms * phi)  # [..., m]

    return jnp.concatenate(
        [
            jnp.sqrt(2) * sin,
            jnp.ones_like(phi),
            jnp.sqrt(2) * cos,
        ],
        axis=-1,
    )[l + m]


def sh_theta(l: int, m: int, theta: float) -> float:
    r"""Theta dependence of spherical harmonics.

    Args:
        lmax: l value
        theta: theta value

    Returns:
        Array of shape ``(l, m)``
    """
    assert theta.ndim == 0
    cos_theta = jnp.cos(theta)
    legendres = e3nn.legendre.legendre(l, cos_theta, phase=1.0, is_normalized=True)  # [l, m, ...]
    sh_theta_comp = legendres[l, jnp.abs(m)]
    return sh_theta_comp


def spherical_harmonic(l: int, m: int) -> float:
    r"""Spherical harmonic (Y_lm)

    Args:
        l: l value
        m: m value

    Returns:
        Returns a function that computes the spherical harmonic for a given theta and phi.
    """

    def Y_lm(theta: float, phi: float) -> float:
        assert theta.shape == phi.shape
        return sh_theta(l, m, theta) * sh_phi(l, m, phi)

    return Y_lm


def fourier_2D(u: int, v: int) -> Callable[[float, float], float]:
    """Fourier function in 2D."""

    def fourier_uv(theta: float, phi: float) -> float:
        return jnp.exp(1j * (u * theta + v * phi)) / (2 * jnp.pi)

    return fourier_uv


@functools.lru_cache(maxsize=None)
def create_spherical_harmonic_signal(l: int, m: int, *, res_theta: int, res_phi: int):
    """Creates a signal for Y^{l,m}."""
    return RectangularSignal.from_function(
        spherical_harmonic(l, m),
        res_theta=res_theta,
        res_phi=res_phi,
        wrap_theta=(m % 2 == 1),
    )


@functools.lru_cache(maxsize=None)
def create_2D_fourier_signal(u: int, v: int, *, res_theta: int, res_phi: int):
    """Creates a signal for Fourier function defined by {u, v}."""
    return RectangularSignal.from_function(fourier_2D(u, v), res_theta=res_theta, res_phi=res_phi, wrap_theta=False)


def to_u_index(u: int, lmax: int) -> int:
    """Returns the index of u in the grid."""
    return u + lmax


def to_v_index(v: int, lmax: int) -> int:
    """Returns the index of v in the grid."""
    return v + lmax


@functools.lru_cache(maxsize=None)
def compute_y(l: int, m: int, u: int, v: int, *, res_theta: int, res_phi: int):
    """Computes y^{l,m}_{u, v}."""
    Y_signal = create_spherical_harmonic_signal(l, m, res_theta=res_theta, res_phi=res_phi)
    F_signal = create_2D_fourier_signal(u, v, res_theta=res_theta, res_phi=res_phi)
    return (Y_signal * F_signal).integrate(area_element="rectangular")


@functools.lru_cache(maxsize=None)
def compute_y_grid(lmax: int, *, res_theta: int, res_phi: int):
    """Computes the grid of y^{l,m}_{u, v}."""
    lm_indices = jnp.arange((lmax + 1) ** 2)
    us = jnp.arange(-lmax, lmax + 1)
    vs = jnp.arange(-lmax, lmax + 1)
    mesh = jnp.meshgrid(lm_indices, us, vs, indexing="ij")

    y_grid = jnp.zeros(((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1), dtype=jnp.complex64)
    for lm_index, u, v in zip(*[m.ravel() for m in mesh]):
        l, m = from_lm_index(lm_index)
        u_index = to_u_index(u, lmax)
        v_index = to_v_index(v, lmax)
        l, m, u, v = int(l), int(m), int(u), int(v)
        y_val = compute_y(l, m, u, v, res_theta=res_theta, res_phi=res_phi)
        y_grid = y_grid.at[lm_index, u_index, v_index].set(y_val)

    assert y_grid.shape == ((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1)
    return y_grid


@functools.lru_cache(maxsize=None)
def compute_z(l: int, m: int, u: int, v: int, *, res_theta: int, res_phi: int):
    """Computes z^{l,m}_{u, v}."""
    Y_signal = create_spherical_harmonic_signal(l, m, res_theta=res_theta, res_phi=res_phi)
    F_signal = create_2D_fourier_signal(u, v, res_theta=res_theta, res_phi=res_phi)
    return (Y_signal * F_signal).integrate(area_element="spherical")


@functools.lru_cache(maxsize=None)
def compute_z_grid(lmax: int, *, res_theta: int, res_phi: int):
    """Computes the grid of z^{l,m}_{u, v}."""
    lm_indices = jnp.arange((lmax + 1) ** 2)
    us = jnp.arange(-lmax, lmax + 1)
    vs = jnp.arange(-lmax, lmax + 1)
    mesh = jnp.meshgrid(lm_indices, us, vs, indexing="ij")

    z_grid = jnp.zeros(((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1), dtype=jnp.complex64)
    for lm_index, u, v in zip(*[m.ravel() for m in mesh]):
        l, m = from_lm_index(lm_index)
        u_index = to_u_index(u, lmax)
        v_index = to_v_index(v, lmax)
        l, m, u, v = int(l), int(m), int(u), int(v)
        z_val = compute_z(l, m, u, v, res_theta=res_theta, res_phi=res_phi)
        z_grid = z_grid.at[lm_index, u_index, v_index].set(z_val)

    assert z_grid.shape == ((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1)
    return z_grid


def convolve_2D_fft(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """2D convolution of x1 and x2 using FFT."""
    convolve_fn = _convolve_2D_fft_single_sample
    for _ in range(x1.ndim - 2):
        convolve_fn = jax.vmap(convolve_fn)
    return convolve_fn(x1, x2)


def _convolve_2D_fft_single_sample(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """2D convolution of x1 and x2 using FFT for a single sample."""
    assert x1.ndim == x2.ndim == 2

    # Get dimensions.
    x1_dim1, x1_dim2 = x1.shape
    x2_dim1, x2_dim2 = x2.shape

    # Calculate full output size.
    full_dim1 = x1_dim1 + x2_dim1 - 1
    full_dim2 = x1_dim2 + x2_dim2 - 1

    # Pad x1 and x2.
    x1_padded = jnp.pad(x1, ((0, full_dim1 - x1_dim1), (0, full_dim2 - x1_dim2)))
    x2_padded = jnp.pad(x2, ((0, full_dim1 - x2_dim1), (0, full_dim2 - x2_dim2)))

    # Perform FFT.
    x1_fft = jnp.fft.fft2(x1_padded)
    x2_fft = jnp.fft.fft2(x2_padded)

    # Multiply in frequency domain.
    result_fft = x1_fft * x2_fft

    # Inverse FFT.
    result = jnp.fft.ifft2(result_fft)
    return result


def convolve_2D_direct(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """2D convolution of x1 and x2 directly."""
    convolve_fn = lambda x1, x2: jax.scipy.signal.convolve2d(x1, x2, mode="full")
    for _ in range(x1.ndim - 2):
        convolve_fn = jax.vmap(convolve_fn)
    return convolve_fn(x1, x2)
