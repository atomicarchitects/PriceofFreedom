"""Implements the Vector Spherical Harmonics (VSH)."""

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import chex


@jax.tree_util.register_pytree_node_class
class VSHCoeffs:
    """Coefficients for the vector spherical harmonics: parity = -1 for VSH, parity = 1 for PVSH."""

    def __init__(self, irreps_array: e3nn.IrrepsArray, parity: int = -1):
        if parity not in [1, -1]:
            raise ValueError(f"Invalid parity {parity}.")

        # Check that the irreps are valid.
        expected_irreps = self.get_vsh_irreps(jmax=irreps_array.irreps.lmax - 1, parity=parity)
        if irreps_array.irreps != expected_irreps:
            raise ValueError(f"Invalid irreps {irreps_array.irreps}, expected {expected_irreps}.")

        self.irreps_array = irreps_array
        self.parity = parity

    def __repr__(self):
        if self.parity == -1:
            vsh_type = "VSH Coefficients"
        elif self.parity == 1:
            vsh_type = "PVSH Coefficients"
        lines = [f"{vsh_type}"]
        for irrep, chunk in zip(self.irreps_array.irreps, self.irreps_array.chunks):
            lines.append(f" {irrep}: {chunk}")
        return "\n".join(lines)

    def jmax(self) -> int:
        """Returns the maximum j in a dictionary of coefficients."""
        return self.irreps_array.irreps.lmax - 1

    def to_irreps_array(self) -> e3nn.IrrepsArray:
        """Converts a dictionary of VSH coefficients to an IrrepsArray."""
        return self.irreps_array

    @staticmethod
    def get_vsh_irreps(*, jmax: int, parity: int) -> e3nn.Irreps:
        """Returns the irreps for the VSH upto some jmax."""
        return get_change_of_basis_matrix(jmax=jmax, parity=parity).irreps

    @classmethod
    def zeros(cls, *, jmax: int, parity: int) -> "VSHCoeffs":
        """Creates a dictionary of all-zeros coefficients for each VSH."""
        return cls(e3nn.zeros(cls.get_vsh_irreps(jmax=jmax, parity=parity)), parity=parity)

    @classmethod
    def normal(cls, *, jmax: int, parity: int, key: chex.PRNGKey) -> "VSHCoeffs":
        """Creates a dictionary of random normal coefficients for each VSH."""
        return cls(
            e3nn.normal(cls.get_vsh_irreps(jmax=jmax, parity=parity), key),
            parity=parity,
        )

    def to_xyz_coeffs(self) -> e3nn.IrrepsArray:
        """Converts a dictionary of VSH coefficients to a 3D IrrepsArray."""
        rtp = get_change_of_basis_matrix(jmax=self.jmax(), parity=self.parity)
        xyz_coeffs = jnp.einsum("ijk,k->ij", rtp.array, self.irreps_array.array)
        xyz_coeffs = e3nn.IrrepsArray(e3nn.s2_irreps(self.jmax()), xyz_coeffs)
        return xyz_coeffs

    def to_vector_signal(self, res_beta: int, res_alpha: int, quadrature: str) -> e3nn.SphericalSignal:
        """Converts a dictionary of VSH coefficients to a vector spherical signal."""
        xyz_coeffs = self.to_xyz_coeffs()
        xyz_coeffs = e3nn.sum(xyz_coeffs.regroup(), axis=-1)
        vector_sig = e3nn.to_s2grid(
            xyz_coeffs,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=quadrature,
            p_val=1,
            p_arg=-1,
            fft=False,
        )
        return vector_sig

    @classmethod
    def from_vector_signal(cls, sig: e3nn.SphericalSignal, jmax: int, parity: int) -> "VSHCoeffs":
        """Returns the components of Y_{j_out, l_out, mj_out} in the signal sig for all mj_out in [-j_out, ..., j_out] and j_out in [-l_out, ..., l_out] and l_out upto lmax."""
        rtp = get_change_of_basis_matrix(jmax=jmax, parity=parity)
        xyz_coeffs = e3nn.from_s2grid(sig, irreps=e3nn.s2_irreps(jmax), fft=False)
        vsh_coeffs = e3nn.IrrepsArray(rtp.irreps, jnp.einsum("ijk,ij->k", rtp.array, xyz_coeffs.array))
        return cls(vsh_coeffs, parity=parity)

    @classmethod
    def vector_spherical_harmonics(cls, j: int, l: int, mj: int, parity: int = -1) -> "VSHCoeffs":
        """Returns a (pseudo)-vector spherical harmonic for a given (j, l, mj)."""
        if j not in [l - 1, l, l + 1]:
            raise ValueError(f"Invalid j={j} for l={l}.")

        if mj not in range(-j, j + 1):
            raise ValueError(f"Invalid mj={mj} for j={j}.")

        irreps = e3nn.Irrep(j, ((-1) ** l) * parity)
        array = jnp.asarray([1.0 if i == mj else 0.0 for i in range(-j, j + 1)])
        irreps_array = e3nn.IrrepsArray(irreps, array)
        irreps_array = irreps_array.extend_with_zeros(cls.get_vsh_irreps(jmax=j, parity=parity))
        coeffs_dict = cls(irreps_array, parity=parity)
        return coeffs_dict

    def reduce_pointwise_cross_product(
        self, other: "VSHCoeffs", res_beta: int, res_alpha: int, quadrature: str
    ) -> "VSHCoeffs":
        """Computes the pointwise cross product on the sphere, and converts back to VSH coefficients."""
        self_sig = self.to_vector_signal(res_beta, res_alpha, quadrature)
        other_sig = other.to_vector_signal(res_beta, res_alpha, quadrature)
        cross_sig = pointwise_cross_product(self_sig, other_sig)
        return VSHCoeffs.from_vector_signal(
            cross_sig,
            jmax=self.jmax() + other.jmax(),
            parity=self.parity * other.parity,
        )

    def reduce_pointwise_dot_product(
        self, other: "VSHCoeffs", res_beta: int, res_alpha: int, quadrature: str
    ) -> e3nn.IrrepsArray:
        """Computes the pointwise dot product on the sphere, and converts back to scalar SH coefficients."""
        self_sig = self.to_vector_signal(res_beta, res_alpha, quadrature)
        other_sig = other.to_vector_signal(res_beta, res_alpha, quadrature)
        dot_sig = pointwise_dot_product(self_sig, other_sig)
        return e3nn.from_s2grid(dot_sig, irreps=e3nn.s2_irreps(self.jmax() + other.jmax()), fft=False)

    def filter(self, lmax: int) -> "VSHCoeffs":
        """Filters out all coefficients with l > lmax."""
        return VSHCoeffs(self.irreps_array.filter(lmax=lmax), parity=self.parity)

    def tree_flatten(self):
        return (self.irreps_array, self.parity), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        irreps_array, parity = aux_data
        return cls(irreps_array, parity=parity)


def get_change_of_basis_matrix(*, jmax: int, parity: int) -> jnp.ndarray:
    """Returns the change of basis matrix."""
    if parity not in [1, -1]:
        raise ValueError(f"Invalid parity {parity}.")

    if jmax == 0:
        jmax = 1

    return e3nn.reduced_tensor_product_basis(
        "ij",
        i=e3nn.Irrep(1, parity),
        j=e3nn.s2_irreps(jmax),
    )


def _wrap_fn_for_vector_signal(fn):
    """vmaps a fn over res_beta and res_alpha axes."""
    fn = jax.vmap(fn, in_axes=-1, out_axes=-1)
    fn = jax.vmap(fn, in_axes=-1, out_axes=-1)
    return fn


def pointwise_cross_product(sig1: e3nn.SphericalSignal, sig2: e3nn.SphericalSignal) -> e3nn.SphericalSignal:
    """Computes the pointwise cross product of two vector signals."""
    return sig1.replace_values(_wrap_fn_for_vector_signal(jnp.cross)(sig1.grid_values, sig2.grid_values))


def pointwise_dot_product(sig1: e3nn.SphericalSignal, sig2: e3nn.SphericalSignal) -> e3nn.SphericalSignal:
    """Computes the pointwise dot product of two vector signals."""
    return sig1.replace_values(_wrap_fn_for_vector_signal(jnp.dot)(sig1.grid_values, sig2.grid_values))
