"""Implements the API for Vector Spherical Harmonics (VSH)."""

from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import chex


@jax.tree_util.register_pytree_node_class
class VSHCoeffs(dict):
    """Parity = -1 for VSH, parity = 1 for PVSH."""

    def __init__(self, parity: int):
        super().__init__()
        if parity not in [1, -1]:
            raise ValueError(f"Invalid parity {parity}.")
        self.parity = parity

    def __setitem__(self, key: Tuple[int, int], value: e3nn.IrrepsArray) -> None:
        j, l = key
        assert (
            value.irreps.num_irreps == 1
        ), f"Invalid count {value.irreps.count} for VSH {j, l}."
        mul, ir = value.irreps[0]
        assert l - 1 <= j <= l + 1, f"Invalid j={j} for VSH {j, l}."
        assert mul == 1, f"Invalid multiplicity {mul} for VSH {j, l}."
        assert ir == VSHCoeffs.get_vsh_irrep(
            j, l, self.parity
        ), f"Invalid irrep {ir} for VSH {j, l} with parity {self.parity}."
        super().__setitem__(key, value)

    def jmax(self) -> int:
        """Returns the maximum j in a dictionary of coefficients."""
        return max(j for j, _ in self.keys())

    def to_irreps_array(self) -> e3nn.IrrepsArray:
        """Converts a dictionary of VSH coefficients to an IrrepsArray."""
        return e3nn.concatenate([v for v in self.values()])

    @staticmethod
    def get_vsh_irrep(j: int, l: int, parity: int) -> e3nn.Irrep:
        """Returns the irrep of a VSH."""
        if parity == -1:
            return e3nn.Irrep(j, (-1) ** (l + 1))
        elif parity == 1:
            return e3nn.Irrep(j, (-1) ** l)
        raise ValueError(f"Invalid parity {parity}.")

    @staticmethod
    def get_vsh_irreps(jmax: int, parity: int) -> e3nn.Irreps:
        """Returns the irreps for the VSH upto some jmax."""
        irreps = []
        for j, l in vsh_iterator(jmax):
            ir = VSHCoeffs.get_vsh_irrep(j, l, parity)
            irreps.append(ir)
        return e3nn.Irreps(irreps)

    @classmethod
    def from_irreps_array(cls, irreps_array: e3nn.IrrepsArray) -> "VSHCoeffs":
        """Converts an IrrepsArray to a dictionary of VSH coefficients."""

        # Try to figure out the parity
        jmax = irreps_array.irreps.lmax
        detected_parity = None
        for parity in [1, -1]:
            if VSHCoeffs.get_vsh_irreps(jmax, parity) == irreps_array.irreps:
                detected_parity = parity
                break

        if detected_parity is None:
            raise ValueError(f"Invalid irreps {irreps_array.irreps} for VSH.")

        coeffs = VSHCoeffs.zeros(jmax, detected_parity)
        for (j, l), (_, ir), chunk in zip(
            coeffs.keys(), irreps_array.irreps, irreps_array.chunks
        ):
            if chunk is None:
                continue
            coeffs[(j, l)] = e3nn.IrrepsArray(ir, chunk[0])

        return coeffs

    def __repr__(self):
        if self.parity == -1:
            vsh_type = "VSH Coefficients"
        elif self.parity == 1:
            vsh_type = "PVSH Coefficients"
        lines = [f"{vsh_type}"]
        for key, value in self.items():
            lines.append(f" {key}: {value}")
        return "\n".join(lines)

    @classmethod
    def zeros(cls, jmax: int, parity: int) -> "VSHCoeffs":
        """Creates a dictionary of all-zeros coefficients for each VSH."""
        coeffs = cls(parity=parity)
        for j, l in vsh_iterator(jmax):
            ir = VSHCoeffs.get_vsh_irrep(j, l, parity)
            coeffs[(j, l)] = e3nn.zeros(ir)
        return coeffs

    @classmethod
    def normal(cls, jmax: int, parity: int, key: chex.PRNGKey) -> "VSHCoeffs":
        """Creates a dictionary of all-zeros coefficients for each VSH."""
        coeffs = cls(parity=parity)
        for j, l in vsh_iterator(jmax):
            ir = VSHCoeffs.get_vsh_irrep(j, l, parity)
            coeffs[(j, l)] = e3nn.normal(ir, key)
            key, _ = jax.random.split(key)
        return coeffs

    def to_vector_coeffs(self) -> e3nn.IrrepsArray:
        """Converts a dictionary of VSH coefficients to a 3D IrrepsArray."""
        rtps = get_change_of_basis_matrices(jmax=self.jmax(), parity=self.parity)
        all_vector_coeffs = []
        for j, l in self.keys():
            rtp = rtps[(j, l)]
            coeff = self[(j, l)]
            if rtp.array.shape[-1] != coeff.array.shape[-1]:
                raise ValueError(
                    f"Invalid shape {coeff.shape} for coefficients with j={j}, l={l}."
                )

            vector_coeffs = jnp.einsum("ijk,k->ij", rtp.array, coeff.array)
            vector_coeffs = e3nn.IrrepsArray(e3nn.s2_irreps(l)[-1], vector_coeffs)
            all_vector_coeffs.append(vector_coeffs)
        return e3nn.concatenate(all_vector_coeffs)

    def to_vector_signal(
        self, res_beta: int, res_alpha: int, quadrature: str
    ) -> e3nn.SphericalSignal:
        """Converts a dictionary of VSH coefficients to a vector spherical signal."""
        vector_coeffs = self.to_vector_coeffs()
        vector_coeffs = e3nn.sum(vector_coeffs.regroup(), axis=-1)
        vector_sig = e3nn.to_s2grid(
            vector_coeffs,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=quadrature,
            p_val=1,
            p_arg=-1,
        )
        return vector_sig

    @classmethod
    def from_vector_signal(
        cls, sig: e3nn.SphericalSignal, jmax: int, parity: int
    ) -> "VSHCoeffs":
        """Returns the components of Y_{j_out, l_out, mj_out} in the signal sig for all mj_out in [-j_out, ..., j_out] and j_out in [-l_out, ..., l_out] and l_out upto lmax."""
        if sig.shape[-3] != 3:
            raise ValueError(f"Invalid shape {sig.shape} for signal.")

        result = VSHCoeffs(parity=parity)
        for j_out, l_out in vsh_iterator(jmax):
            result[(j_out, l_out)] = get_vsh_coeffs_at_j(sig, j_out, l_out, parity)
        return result

    def filter(self, keep: Sequence[e3nn.Irreps]) -> "VSHCoeffs":
        """Filters out to keep only certain irreps."""
        keep = e3nn.Irreps(keep)
        new_coeffs = VSHCoeffs(parity=self.parity)
        for (j, l), coeff in self.items():
            _, coeff_ir = coeff.irreps[0]
            if coeff_ir in keep:
                new_coeffs[(j, l)] = coeff
        return new_coeffs

    @classmethod
    def vector_spherical_harmonics(
        cls, j: int, l: int, mj: int, parity: int = -1
    ) -> "VSHCoeffs":
        """Returns a (pseudo)-vector spherical harmonic for a given (j, l, mj)."""
        if j not in [l - 1, l, l + 1]:
            raise ValueError(f"Invalid j={j} for l={l}.")

        if mj not in range(-j, j + 1):
            raise ValueError(f"Invalid mj={mj} for j={j}.")

        coeffs = e3nn.IrrepsArray(
            VSHCoeffs.get_vsh_irrep(j, l, parity),
            jnp.asarray([1.0 if i == mj else 0.0 for i in range(-j, j + 1)]),
        )
        coeffs_dict = cls(parity=parity)
        coeffs_dict[(j, l)] = coeffs
        return coeffs_dict

    def reduce_pointwise_cross_product(
        self, other: "VSHCoeffs", res_beta: int, res_alpha: int, quadrature: str
    ) -> "VSHCoeffs":
        """Computes the pointwise cross product on the sphere, and converts back to VSH coefficients."""
        self_sig = self.to_vector_signal(res_beta, res_alpha, quadrature)
        other_sig = other.to_vector_signal(res_beta, res_alpha, quadrature)
        cross_sig = cross_product(self_sig, other_sig)
        return VSHCoeffs.from_vector_signal(
            cross_sig,
            jmax=self.jmax() + other.jmax(),
            parity=self.parity * other.parity,
        )

    def reduce_pointwise_dot_product(
        self, other: "VSHCoeffs", res_beta: int, res_alpha: int, quadrature: str
    ) -> e3nn.SphericalSignal:
        """Computes the pointwise dot product on the sphere, and converts back to scalar SH coefficients."""
        self_sig = self.to_vector_signal(res_beta, res_alpha, quadrature)
        other_sig = other.to_vector_signal(res_beta, res_alpha, quadrature)
        dot_sig = dot_product(self_sig, other_sig)
        return e3nn.from_s2grid(
            dot_sig,
            irreps=e3nn.s2_irreps(self.jmax() + other.jmax()),
        )

    def tree_flatten(self):
        return list(self.values()), (self.keys(), self.parity)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        keys, parity = aux_data
        values = children
        coeffs = cls(parity=parity)
        coeffs.update(zip(keys, values))
        return coeffs


def get_vsh_coeffs_at_mj(
    sig: e3nn.SphericalSignal, j_out: int, l_out: int, mj_out: int
) -> float:
    """Returns the component of Y_{j_out, l_out, mj_out} in the signal sig."""
    vsh_coeff = VSHCoeffs.vector_spherical_harmonics(j_out, l_out, mj_out)
    vsh_signal = vsh_coeff.to_vector_signal(
        res_beta=sig.res_beta, res_alpha=sig.res_alpha, quadrature=sig.quadrature
    )
    sig_vsh_dot_product = dot_product(sig, vsh_signal)
    return sig_vsh_dot_product.integrate().array[-1] / (4 * jnp.pi)


def vsh_iterator(jmax: int):
    """Iterates over all VSH up to some jmax."""
    for j in range(jmax + 1):
        for l in [j - 1, j, j + 1]:
            if j == 0 and l != 1:
                continue
            yield j, l


def get_change_of_basis_matrices(jmax: int, parity: int) -> jnp.ndarray:
    """Returns the change of basis for each (j, l) pair."""
    if parity not in [1, -1]:
        raise ValueError(f"Invalid parity {parity}.")

    rtps = {}
    for j, l in vsh_iterator(jmax):
        rtp = e3nn.reduced_tensor_product_basis(
            "ij",
            i=e3nn.Irrep(1, parity),
            j=e3nn.Irrep(l, (-1) ** (l)),
            keep_ir=VSHCoeffs.get_vsh_irrep(j, l, parity),
        )
        rtps[(j, l)] = rtp
    return rtps


def get_vsh_coeffs_at_j(
    sig: e3nn.SphericalSignal,
    j_out: int,
    l_out: int,
    parity_out: int,
) -> e3nn.IrrepsArray:
    """Returns the components of Y_{j_out, l_out, mj_out} in the signal sig for all mj_out in [-j_out, ..., j_out]."""
    computed_coeffs = jnp.stack(
        [
            get_vsh_coeffs_at_mj(sig, j_out, l_out, mj_out)
            for mj_out in range(-j_out, j_out + 1)
        ]
    )
    computed_coeffs = e3nn.IrrepsArray(
        VSHCoeffs.get_vsh_irrep(j_out, l_out, parity_out), computed_coeffs
    )
    return computed_coeffs



def _wrap_fn_for_vector_signal(fn):
    """vmaps a fn over res_beta and res_alpha axes."""
    fn = jax.vmap(fn, in_axes=-1, out_axes=-1)
    fn = jax.vmap(fn, in_axes=-1, out_axes=-1)
    return fn


def cross_product(
    sig1: e3nn.SphericalSignal, sig2: e3nn.SphericalSignal
) -> e3nn.SphericalSignal:
    """Computes the pointwise cross product of two vector signals."""
    return sig1.replace_values(
        _wrap_fn_for_vector_signal(jnp.cross)(sig1.grid_values, sig2.grid_values)
    )


def dot_product(
    sig1: e3nn.SphericalSignal, sig2: e3nn.SphericalSignal
) -> e3nn.SphericalSignal:
    """Computes the pointwise dot product of two vector signals."""
    return sig1.replace_values(
        _wrap_fn_for_vector_signal(jnp.dot)(sig1.grid_values, sig2.grid_values)
    )
