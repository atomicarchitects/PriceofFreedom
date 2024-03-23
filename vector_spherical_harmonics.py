"""Implements the API for Vector Spherical Harmonics (VSH)."""

from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import chex


def vsh_iterator(jmax: int):
    """Iterates over all VSH up to some jmax."""
    for j in range(jmax + 1):
        for l in [j - 1, j, j + 1]:
            if j == 0 and l != 1:
                continue
            yield j, l


def get_vsh_irrep(j: int, l: int, parity: int) -> e3nn.Irrep:
    """Returns the irrep of a VSH."""
    if parity == -1:
        return e3nn.Irrep(j, (-1) ** (l + 1))
    elif parity == 1:
        return e3nn.Irrep(j, (-1) ** l)
    raise ValueError(f"Invalid parity {parity}.")


def get_vsh_irreps(jmax: int, parity: int) -> e3nn.Irreps:
    """Returns the irreps for the VSH upto some jmax."""
    irreps = []
    for j, l in vsh_iterator(jmax):
        ir = get_vsh_irrep(j, l, parity)
        irreps.append(ir)
    return e3nn.Irreps(irreps)


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
            keep_ir=get_vsh_irrep(j, l, parity),
        )
        rtps[(j, l)] = rtp
    return rtps


class VSHCoeffs(dict):
    def __init__(self, parity, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        assert ir == get_vsh_irrep(
            j, l, self.parity
        ), f"Invalid irrep {ir} for VSH {j, l}."
        super().__setitem__(key, value)

    def get_jmax(self) -> int:
        """Returns the maximum j in a dictionary of coefficients."""
        return max(j for j, _ in self.keys())

    def to_irreps_array(self) -> e3nn.IrrepsArray:
        """Converts a dictionary of VSH coefficients to an IrrepsArray."""
        return e3nn.concatenate([v for v in self.values()])

    @classmethod
    def from_irreps_array(cls, irreps_array: e3nn.IrrepsArray) -> "VSHCoeffs":
        """Converts an IrrepsArray to a dictionary of VSH coefficients."""

        # Try to figure out the parity
        jmax = irreps_array.irreps.lmax
        detected_parity = None
        for parity in [1, -1]:
            if get_vsh_irreps(jmax, parity) == irreps_array.irreps:
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
        lines = [f"VSHCoeffs(parity={self.parity})"]
        for key, value in self.items():
            lines.append(f" {key}: {value}")
        return "\n".join(lines)

    @classmethod
    def zeros(cls, jmax: int, parity: int) -> "VSHCoeffs":
        """Creates a dictionary of all-zeros coefficients for each VSH."""
        coeffs = cls(parity=parity)
        for j, l in vsh_iterator(jmax):
            ir = get_vsh_irrep(j, l, parity)
            coeffs[(j, l)] = e3nn.zeros(ir)
        return coeffs

    @classmethod
    def normal(cls, jmax: int, parity: int, key: chex.PRNGKey) -> "VSHCoeffs":
        """Creates a dictionary of all-zeros coefficients for each VSH."""
        coeffs = cls(parity=parity)
        for j, l in vsh_iterator(jmax):
            ir = get_vsh_irrep(j, l, parity)
            coeffs[(j, l)] = e3nn.normal(ir, key)
            key, _ = jax.random.split(key)
        return coeffs

    def to_vector_coeffs(self) -> e3nn.IrrepsArray:
        """Converts a dictionary of VSH coefficients to a 3D IrrepsArray."""
        rtps = get_change_of_basis_matrices(jmax=self.get_jmax(), parity=self.parity)
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
        self, res_beta: int = 90, res_alpha: int = 89, quadrature="soft"
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

    def filter(self, keep: Sequence[e3nn.Irreps]) -> "VSHCoeffs":
        """Filters out to keep only certain irreps."""
        keep = e3nn.Irreps(keep)
        new_coeffs = VSHCoeffs(parity=self.parity)
        for (j, l), coeff in self.items():
            _, coeff_ir = coeff.irreps[0]
            if coeff_ir in keep:
                new_coeffs[(j, l)] = coeff
        return new_coeffs
