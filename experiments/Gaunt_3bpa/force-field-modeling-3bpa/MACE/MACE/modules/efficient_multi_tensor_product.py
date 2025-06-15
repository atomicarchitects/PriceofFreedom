from typing import Optional
import os

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from e3nn import o3
import e3nn.util.jit
import e3nn.util.test
from .vector_spherical_harmonics import VectorSphericalHarmonics
from .efficient_utils import FFT_batch_channel, sh2f_batch_channel, f2sh_batch_channel

current_directory = os.path.dirname(os.path.abspath(__file__))
sh2f_bases_dict = torch.load(os.path.join(current_directory, "coefficient_sh2f.pt"))
f2sh_bases_dict = torch.load(os.path.join(current_directory, "coefficient_f2sh.pt"))


def debug(*args):
    pass
    # print(*args)


class EfficientMultiTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        device: str,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channels = irreps_in.count((0, 1))
        self.num_elements = num_elements
        self.irreps_in_per_channel = o3.Irreps(
            [(mul // self.num_channels, ir) for mul, ir in self.irreps_in]
        )
        self.irreps_out_per_channel = o3.Irreps(
            [(mul // self.num_channels, ir) for mul, ir in self.irreps_out]
        )
        del irreps_in, irreps_out
        self.correlation = correlation
        self.device = device
        # debug(
        #     "bro",
        #     self.irreps_in,
        #     self.irreps_out,
        #     self.num_channels,
        #     self.correlation,
        #     num_elements,
        # )

        L_in = self.irreps_in.lmax + 1
        L_out = self.irreps_out.lmax + 1
        self.L_in = L_in
        self.L_out = L_out

        self.sh2f_bases = sh2f_bases_dict[L_in].to(device)
        lmaxs = torch.arange(2, correlation + 1) * (L_in - 1)
        self.f2sh_bases_list = list(
            map(lambda lmax: f2sh_bases_dict[lmax + 1].to(device), lmaxs.tolist())
        )
        self.offsets_st = lmaxs - L_out + 1
        self.offsets_ed = lmaxs + L_out

        def gen_mask(L):
            left_indices = torch.arange(L, device=device).view(1, -1)
            right_indices = torch.arange(L - 2, -1, -1, device=device).view(1, -1)
            column_indices = torch.cat((left_indices, right_indices), dim=1).repeat(
                L, 1
            )
            row_indices = (
                torch.arange(L, device=device).view(-1, 1).repeat(1, 2 * L - 1)
            )
            mask = torch.abs(column_indices - (L - 1)) <= row_indices
            mask2D = (torch.ones(L, 2 * L - 1, device=device) * mask).to(bool)
            return mask2D.flatten()

        self.mask_i, self.mask_o = list(map(gen_mask, [L_in, L_out]))

        slices = 2 * torch.arange(L_out, device=device) + 1
        self.slices = slices.tolist()

        self.weights = nn.ParameterDict({})
        for i in range(1, correlation + 1):
            w = nn.Parameter(
                torch.randn(1, num_elements, self.num_channels, self.L_out)
            )
            self.weights[str(i)] = w
        self.equivarianced_checked = False

    def forward(self, atom_feat: torch.tensor, atom_type: torch.Tensor):
        if not self.equivarianced_checked:
            self.equivarianced_checked = True
            e3nn.util.test.assert_equivariant(
                self.forward,
                args_in=[atom_feat, atom_type],
                irreps_in=[self.irreps_in_per_channel, f"{self.num_elements}x0e"],
                irreps_out=[self.irreps_out],
            )

        # debug("irreps_in", self.irreps_in, self.irreps_in.dim)
        # debug("inputs", atom_feat.shape, atom_type.shape)

        # inputs are of shape:
        # atom_feat: (B, C, self.irreps_in_per_channel.dim)
        # atom_type: (B, self.num_elements)

        # self.weights[i] are all of shape: (1, num_elements, C, num_output_irreps)
        # atom_type.unsqueeze(-1).unsqueeze(-1) is of shape: (B, num_elements, 1, 1)
        # The product of the two is of shape: (B, num_elements, C, num_output_irreps)
        # weights is of shape: (B, C, num_output_irreps, 1) after summing over num_elements
        # feat4D is of shape: (B, C, L_in, 2L_in-1), which gets sliced to (B, C, num_output_irreps, 2num_output_irreps-1)

        # Convert from 3D to 4D, so as to facilitate the implementation of Efficient Gaunt TP
        # The time taken by this convert step is minimal
        n_nodes = atom_feat.shape[0]
        feat3D = torch.zeros(
            n_nodes, self.num_channels, self.mask_i.shape[0], device=self.device
        )
        feat3D[:, :, self.mask_i] = atom_feat
        feat4D = feat3D.reshape(
            n_nodes, self.num_channels, self.L_in, -1
        )  # (B, C, L_in, 2L_in-1)

        # debug(atom_type[:5])
        # debug("feat3D", feat3D.shape)
        # debug("feat4D", feat4D.shape)
        # for w in self.weights.values():
        #     debug(w.shape, atom_type.unsqueeze(-1).unsqueeze(-1).shape)

        # @T.C.: Perform Efficient Gaunt TP
        weights = (
            (self.weights["1"] * atom_type.unsqueeze(-1).unsqueeze(-1))
            .sum(1)
            .unsqueeze(-1)
        )  # (B, C, L_out, 1)
        # debug(
        #     "weights",
        #     (self.weights["1"] * atom_type.unsqueeze(-1).unsqueeze(-1)).shape,
        #     weights.shape,
        # )
        result = (
            feat4D[
                :, :, : self.L_out, self.L_in - self.L_out : self.L_in + self.L_out - 1
            ]
            * weights
        )

        fs_out = {}
        fs_out[1] = sh2f_batch_channel(feat4D, self.sh2f_bases)
        for nu in range(2, self.correlation + 1):
            if nu % 2 == 0:
                fs_out[nu] = FFT_batch_channel(fs_out[nu // 2], fs_out[nu // 2])
            else:
                fs_out[nu] = FFT_batch_channel(fs_out[nu // 2], fs_out[nu // 2 + 1])
            idx = nu - 2
            weights = (
                (self.weights[str(nu)] * atom_type.unsqueeze(-1).unsqueeze(-1))
                .sum(1)
                .unsqueeze(-1)
            )
            result += (
                weights
                * f2sh_batch_channel(fs_out[nu], self.f2sh_bases_list[idx]).real[
                    :, :, : self.L_out, self.offsets_st[idx] : self.offsets_ed[idx]
                ]
            )

        # Convert from 4D to 2D, so as to match the original codebase
        # The time taken by this convert step is minimal
        if self.L_out == 1:
            return result.squeeze()
        else:
            result3D_unfiltered = result.reshape(n_nodes, self.num_channels, -1)
            result3D = torch.zeros(
                n_nodes, self.num_channels, self.mask_o.shape[0], device=self.device
            )
            result3D = result3D_unfiltered[:, :, self.mask_o]
            irreps = torch.split(result3D, self.slices, dim=-1)
            irreps_flatten = list(map(lambda x: x.flatten(start_dim=1), irreps))
            result2D = torch.cat(irreps_flatten, dim=-1)
            return result2D


@e3nn.util.jit.compile_mode("trace")
class GauntTensorProductS2Grid(nn.Module):
    """S2 grid version of GauntTensorProduct."""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        res_beta: Optional[int] = None,
        res_alpha: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        lmax = max(self.irreps_in1.lmax, self.irreps_in2.lmax, self.irreps_out.lmax)
        if res_beta is None or res_alpha is None:
            res_beta = 2 * (lmax + 1)
            res_alpha = 2 * lmax + 1

        self.to_s2grid1 = o3.ToS2Grid(
            lmax=self.irreps_in1.lmax,
            res=(res_beta, res_alpha),
            device=device,
        )
        self.to_s2grid2 = o3.ToS2Grid(
            lmax=self.irreps_in2.lmax,
            res=(res_beta, res_alpha),
            device=device,
        )
        self.from_s2grid = o3.FromS2Grid(
            lmax=self.irreps_out.lmax,
            res=(res_beta, res_alpha),
            device=device,
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # debug("gs2grid")
        # debug("inputs", input1.shape, input2.shape)
        input1_grid = self.to_s2grid1(input1)
        input2_grid = self.to_s2grid2(input2)
        # debug("inputs on grid", input1_grid.shape, input2_grid.shape)
        output_grid = input1_grid * input2_grid
        # debug("output on grid", output_grid.shape)
        output = self.from_s2grid(output_grid)
        # debug("output", output.shape)
        # debug("gs2grid done")
        # debug()
        return output


@e3nn.util.jit.compile_mode("trace")
class ChannelCombiner(nn.Module):
    """Combines channels of a tensor by reshaping."""

    def __init__(self, num_channels_to_combine: int):
        """Combines channels of a tensor by reshaping."""

        super().__init__()
        self.num_channels_to_combine = num_channels_to_combine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            x.shape[:-2]
            + (
                x.shape[-2] // self.num_channels_to_combine,
                self.num_channels_to_combine * x.shape[-1],
            )
        )
        return x

    def uncombine(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            x.shape[:-2]
            + (
                x.shape[-2] * self.num_channels_to_combine,
                x.shape[-1] // self.num_channels_to_combine,
            )
        )
        return x


def debug_irreps_array(input, irreps):
    for ir, slicer in zip(irreps, irreps.slices()):
        print(
            ir,
            torch.min(input[..., slicer]).detach().cpu().numpy().round(3),
            torch.mean(input[..., slicer]).detach().cpu().numpy().round(3),
            torch.max(input[..., slicer]).detach().cpu().numpy().round(3),
        )

@e3nn.util.jit.compile_mode("trace")
class VectorGauntTensorProductS2Grid(nn.Module):
    """S2 grid version of GauntTensorProduct."""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        res_beta: Optional[int] = None,
        res_alpha: Optional[int] = None,
        device: Optional[torch.device] = None,
        apply_activation: bool = False,
    ):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        lmax = max(self.irreps_in1.lmax, self.irreps_in2.lmax, self.irreps_out.lmax)
        if res_beta is None or res_alpha is None:
            res_beta = 2 * (lmax + 1)
            res_alpha = 2 * lmax + 1

        self.scalar_interaction = True
        
        self.vsh1 = VectorSphericalHarmonics(
            lmax=self.irreps_in1.lmax - 1,
            res_beta=res_beta,
            res_alpha=res_alpha,
            parity=-1,
            scalar_interaction=self.scalar_interaction,
            device=device,
        )
        self.linear1 = e3nn.o3.Linear(
            irreps_in=self.irreps_in1, irreps_out=self.vsh1.irreps
        )

        self.vsh2 = VectorSphericalHarmonics(
            lmax=self.irreps_in2.lmax - 1,
            res_beta=res_beta,
            res_alpha=res_alpha,
            parity=-1,
            scalar_interaction=self.scalar_interaction,
            device=device,
        )
        self.linear2 = e3nn.o3.Linear(
            irreps_in=self.irreps_in2, irreps_out=self.vsh2.irreps
        )

        if apply_activation:
            self.activation = nn.SELU()
        else:
            self.activation = None

        self.pvsh = VectorSphericalHarmonics(
            lmax=self.irreps_out.lmax - 1,
            res_beta=res_beta,
            res_alpha=res_alpha,
            parity=+1,
            scalar_interaction=self.scalar_interaction,
            device=device,
        )
        self.linear_out = e3nn.o3.Linear(
            irreps_in=self.pvsh.irreps, irreps_out=self.irreps_out
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = self.linear1(input1)
        input2 = self.linear2(input2)

        # print("input1")
        # debug_irreps_array(input1, self.linear1.irreps_out)

        # print("input2")
        # debug_irreps_array(input2, self.linear2.irreps_out)

        input1_signal = self.vsh1.to_vector_signal(input1)
        input2_signal = self.vsh2.to_vector_signal(input2)

        if self.scalar_interaction:
            output_signal = torch.concatenate(
                [
                    torch.mul(input1_signal[..., :1, :, :], input2_signal[..., :1, :, :]),
                    torch.cross(
                        input1_signal[..., 1:, :, :], input2_signal[..., 1:, :, :], dim=-3
                    ),
                ],
                dim=-3,
            )
        else:
            output_signal = torch.cross(input1_signal, input2_signal, dim=-3)

        if self.activation is not None:
            output_signal = torch.concatenate(
                [
                    output_signal[..., :1, :, :],
                    torch.mul(self.activation(output_signal[..., :1, :, :]), output_signal[..., 1:, :, :]),
                ],
                dim=-3,
            )

        output = self.pvsh.from_vector_signal(output_signal)

        # Scale factor due to normalization of VGTP.
        output *= 100

        # print("output")
        # debug_irreps_array(output, self.pvsh.irreps)

        output = self.linear_out(output)
        return output


class EfficientMultiTensorProductS2Grid(nn.Module):
    """Efficient Multi-Tensor Product with S2 Grid."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        device: str,
        use_vector_spherical_harmonics: bool,
        num_channels_to_combine_vsh: int,
    ):
        """Efficient Multi-Tensor Product with S2 Grid."""
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channels = irreps_in.count((0, 1))
        self.num_elements = num_elements
        self.irreps_in_per_channel = o3.Irreps(
            [(mul // self.num_channels, ir) for mul, ir in self.irreps_in]
        )
        self.irreps_out_per_channel = o3.Irreps(
            [(mul // self.num_channels, ir) for mul, ir in self.irreps_out]
        )
        del irreps_in, irreps_out
        self.correlation = correlation
        self.device = device

        L_in = self.irreps_in.lmax + 1
        L_out = self.irreps_out.lmax + 1
        self.L_in = L_in
        self.L_out = L_out

        if use_vector_spherical_harmonics:
            self.num_channels_to_combine = num_channels_to_combine_vsh
            assert self.num_channels % self.num_channels_to_combine == 0
        else:
            self.num_channels_to_combine = 1

        self.num_channels_after_combine = (
            self.num_channels // self.num_channels_to_combine
        )

        self.weights = nn.ParameterDict({})
        self.weight_repeats = torch.concatenate(
            [
                torch.as_tensor(ir.dim).repeat(mul)
                for mul, ir in self.irreps_out_per_channel
            ]
        ).to(device)
        # debug("weight_repeats", self.weight_repeats)

        if use_vector_spherical_harmonics:
            self.channel_combiner = ChannelCombiner(
                num_channels_to_combine=self.num_channels_to_combine,
            )
        else:
            self.channel_combiner = None

        for i in range(1, correlation + 1):
            w = nn.Parameter(
                torch.randn(
                    1,
                    self.num_elements,
                    self.num_channels,
                    self.L_out,
                )
            )
            self.weights[str(i)] = w

        if use_vector_spherical_harmonics:
            tp = VectorGauntTensorProductS2Grid
        else:
            tp = GauntTensorProductS2Grid

        self.tps = nn.ModuleDict({})
        input_lmax = self.irreps_in_per_channel.lmax

        def get_irreps(nu):
            ir = o3.Irreps.spherical_harmonics(
                nu * input_lmax,
                p=-1,
            )
            ir += o3.Irreps.spherical_harmonics(
                nu * input_lmax,
                p=+1,
            )
            return ir

        for nu in range(1, correlation):
            self.tps[str(nu)] = tp(
                irreps_in1=self.num_channels_to_combine * get_irreps(nu),
                irreps_in2=self.num_channels_to_combine * get_irreps(1),
                irreps_out=self.num_channels_to_combine * get_irreps(nu + 1),
                device=device,
            )

        self.slices = 2 * torch.arange(L_out, device=device) + 1
        self.slices = self.slices.tolist()
        self.equivarianced_checked = True

    def forward(self, atom_feat: torch.tensor, atom_type: torch.Tensor):
        if not self.equivarianced_checked:
            self.equivarianced_checked = True
            e3nn.util.test.assert_equivariant(
                self.forward,
                args_in=[atom_feat, atom_type],
                irreps_in=[self.irreps_in_per_channel, f"{self.num_elements}x0e"],
                irreps_out=[self.irreps_out],
            )

        if self.channel_combiner is not None:
            atom_feat = self.channel_combiner(atom_feat)
        # atom_feat: (B, C, )
        # debug("inputs", atom_feat.shape, atom_type.shape)

        # Perform Efficient Gaunt TP
        batch_size = atom_feat.shape[0]
        result = torch.zeros(
            (
                batch_size,
                self.num_channels,
                self.irreps_out_per_channel.dim,
            ),
            device=self.device,
        )
        # debug("result", result.shape, self.irreps_out_per_channel)
        # debug("irreps_out_per_channel", self.irreps_out_per_channel.dim)
        prod = atom_feat

        for nu in range(1, self.correlation + 1):
            # Compute weights for this iteration.
            weights = (
                self.weights[str(nu)] * atom_type.unsqueeze(-1).unsqueeze(-1)
            ).sum(1)
            # The weights are of shape: (B, C, self.irreps_out_per_channel.num_irreps)
            # Repeat the weights for each dimension of the output irreps.
            weights = weights.repeat_interleave(self.weight_repeats, dim=-1)
            # The weights are now of shape: (B, C, self.irreps_out_per_channel.output_dim)

            # Mix in weights, and add to current result.
            if self.channel_combiner is not None:
                prod_reshaped = self.channel_combiner.uncombine(prod)
            else:
                prod_reshaped = prod
            prod_reshaped = prod_reshaped[..., :result.shape[-1]]
            result += weights * prod_reshaped

            # Perform product.
            if nu < self.correlation:
                prod = self.tps[str(nu)](prod, atom_feat)

        # Uncombine channels.
        # debug("result final", result.shape, self.irreps_out_per_channel)

        # Convert to 2D.
        irreps = torch.split(result, self.slices, dim=-1)
        irreps_flatten = list(map(lambda x: x.flatten(start_dim=1), irreps))
        result2D = torch.cat(irreps_flatten, dim=-1)
        return result2D
