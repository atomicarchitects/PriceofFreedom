from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import e3nn


def get_change_of_basis_matrix(lmax: int, scalar_interaction: bool, parity: int) -> Tuple[torch.Tensor, e3nn.o3.Irreps]:
    """Gets change of basis matrix for vector spherical harmonics."""
    ir_on_s2 = e3nn.o3.Irreps([(1, (1, parity))])
    if scalar_interaction:
        ir_on_s2 = e3nn.o3.Irreps("0e") + ir_on_s2
    tp = e3nn.o3.ReducedTensorProducts("ij", i=ir_on_s2, j=e3nn.o3.Irreps.spherical_harmonics(lmax))

    return tp.change_of_basis, tp.irreps_out


class VectorSphericalHarmonics(nn.Module):
    """Barebones implementation of the vector spherical harmonics."""

    def __init__(self, lmax: int, res_beta: int, res_alpha: int, parity: Union[int, str], scalar_interaction: bool, device: Optional[torch.device] = None) -> None:
        """Parity -1 for VSH, parity +1 for PVSH."""
        super().__init__()

        if isinstance(parity, str):
            if parity == "pvsh":
                parity = 1
            elif parity == "vsh":
                parity = -1
            else:
                raise ValueError(f"Invalid parity: {parity}")

        self.lmax = lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha

        self.rtp, self.irreps = get_change_of_basis_matrix(lmax=lmax, scalar_interaction=scalar_interaction, parity=parity)
        self.rtp = self.rtp.to(device)
        self.to_s2grid = e3nn.o3.ToS2Grid(lmax=lmax, res=(res_beta, res_alpha), device=device)
        self.from_s2grid = e3nn.o3.FromS2Grid(lmax=lmax, res=(res_beta, res_alpha), device=device)

    def to_vector_signal(self, vsh_coeffs: torch.Tensor) -> torch.Tensor:
        xyz_coeffs = torch.einsum("ijk, ...i -> ...jk", self.rtp, vsh_coeffs)
        vector_signal = self.to_s2grid(xyz_coeffs)
        return vector_signal

    def from_vector_signal(self, vector_signal: torch.Tensor) -> torch.Tensor:
        xyz_coeffs = self.from_s2grid(vector_signal)
        vsh_coeffs = torch.einsum("ijk, ...jk -> ...i", self.rtp, xyz_coeffs)
        return vsh_coeffs