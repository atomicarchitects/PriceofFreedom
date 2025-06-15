"""Parameterized tensor products for use in neural networks."""
from typing import Optional

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax.linen as nn

from freedom.tensor_products import functional
from freedom.tensor_products.vector_spherical_harmonics import VSHCoeffs


class ClebschGordanTensorProductDense(nn.Module):
    """Dense version of Clebsch-Gordan tensor product."""

    irrep_normalization: str
    apply_output_linear: bool

    @nn.compact
    def __call__(
        self,
        input1: e3nn.IrrepsArray,
        input2: e3nn.IrrepsArray,
        *,
        filter_ir_out=None,
    ) -> e3nn.IrrepsArray:
        output = functional.clebsch_gordan_tensor_product_dense(
            input1,
            input2,
            filter_ir_out=filter_ir_out,
            irrep_normalization=self.irrep_normalization,
        )
        if self.apply_output_linear:
            output = e3nn.flax.Linear(output.irreps)(output)
        return output


class ClebschGordanTensorProductSparse(nn.Module):
    """Sparse version of Clebsch-Gordan tensor product."""

    irrep_normalization: str
    apply_output_linear: bool

    @nn.compact
    def __call__(
        self,
        input1: e3nn.IrrepsArray,
        input2: e3nn.IrrepsArray,
        *,
        filter_ir_out=None,
    ) -> e3nn.IrrepsArray:
        output = functional.clebsch_gordan_tensor_product_sparse(
            input1,
            input2,
            filter_ir_out=filter_ir_out,
            irrep_normalization=self.irrep_normalization,
        )
        if self.apply_output_linear:
            output = e3nn.flax.Linear(output.irreps)(output)

        return output


class GauntTensorProductAllParitiesS2Grid(nn.Module):
    """Gaunt tensor product concatenated over all parities, using signals on S2."""

    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str
    lmax_grid: bool = True

    @nn.compact
    def __call__(self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # Taking the max lmax out of the 2 inputs for the grid
        lmax = max(input1.irreps.lmax, input2.irreps.lmax)

        tp1 = GauntTensorProductS2Grid(
            p_val1=1,
            p_val2=1,
            num_channels=self.num_channels,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp2 = GauntTensorProductS2Grid(
            p_val1=1,
            p_val2=-1,
            num_channels=self.num_channels,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp3 = GauntTensorProductS2Grid(
            p_val1=-1,
            p_val2=1,
            num_channels=self.num_channels,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class GauntTensorProductS2Grid(nn.Module):
    """Gaunt tensor product using signals on S2."""

    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str
    p_val1: Optional[int] = None
    p_arg1: Optional[int] = None
    p_val2: Optional[int] = None
    p_arg2: Optional[int] = None

    @nn.compact
    def __call__(self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # Project the inputs to the desired parity and channels.
        p_val1, p_arg1 = functional.get_parities(input1.irreps, self.p_val1, self.p_arg1)
        input1_c = e3nn.flax.Linear(
            e3nn.s2_irreps(input1.irreps.lmax, p_val=p_val1, p_arg=p_arg1) * self.num_channels,
            force_irreps_out=True,
            name="linear_in1",
        )(input1)
        input1_c = input1_c.mul_to_axis(self.num_channels)

        p_val2, p_arg2 = functional.get_parities(input2.irreps, self.p_val2, self.p_arg2)
        input2_c = e3nn.flax.Linear(
            e3nn.s2_irreps(input2.irreps.lmax, p_val=p_val2, p_arg=p_arg2) * self.num_channels,
            force_irreps_out=True,
            name="linear_in2",
        )(input2)
        input2_c = input2_c.mul_to_axis(self.num_channels)

        # Compute the tensor product.
        output_c = functional.gaunt_tensor_product_s2grid(
            input1_c,
            input2_c,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val1=p_val1,
            p_arg1=p_arg1,
            p_val2=p_val2,
            p_arg2=p_arg2,
            s2grid_fft=False,
        )

        # Expand the channel dimension in the output.
        output_c = output_c.axis_to_mul()
        output_c = e3nn.flax.Linear(output_c.irreps, name="linear_out")(output_c)
        return output_c


class GauntTensorProductAllParities2DFourier(nn.Module):
    """Gaunt tensor product concatenated over all parities, 2D Fourier functions."""

    num_channels: int
    res_theta: int
    res_phi: int
    convolution_type: str

    @nn.compact
    def __call__(self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        tp1 = GauntTensorProduct2DFourier(
            num_channels=self.num_channels,
            p_val1=1,
            p_val2=1,
            res_theta=self.res_theta,
            res_phi=self.res_phi,
            convolution_type=self.convolution_type,
        )(input1, input2)
        tp2 = GauntTensorProduct2DFourier(
            num_channels=self.num_channels,
            p_val1=1,
            p_val2=-1,
            res_theta=self.res_theta,
            res_phi=self.res_phi,
            convolution_type=self.convolution_type,
        )(input1, input2)
        tp3 = GauntTensorProduct2DFourier(
            num_channels=self.num_channels,
            p_val1=-1,
            p_val2=1,
            res_theta=self.res_theta,
            res_phi=self.res_phi,
            convolution_type=self.convolution_type,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class GauntTensorProduct2DFourier(nn.Module):
    """Gaunt tensor product using 2D Fourier functions."""

    p_val1: int
    p_val2: int
    num_channels: int
    res_theta: int
    res_phi: int
    convolution_type: str

    @nn.compact
    def __call__(self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # Project the inputs to the desired parity and channels.
        input1_c = e3nn.flax.Linear(
            e3nn.s2_irreps(input1.irreps.lmax, p_val=self.p_val1, p_arg=-1) * self.num_channels,
            force_irreps_out=True,
            name="linear_in1",
        )(input1)
        input1_c = input1_c.mul_to_axis(self.num_channels)

        input2_c = e3nn.flax.Linear(
            e3nn.s2_irreps(input2.irreps.lmax, p_val=self.p_val2, p_arg=-1) * self.num_channels,
            force_irreps_out=True,
            name="linear_in2",
        )(input2)
        input2_c = input2_c.mul_to_axis(self.num_channels)

        # Compute the tensor product.
        output_c = functional.gaunt_tensor_product_2D_fourier(
            input1_c,
            input2_c,
            res_theta=self.res_theta,
            res_phi=self.res_phi,
            convolution_type=self.convolution_type,
        )

        # Expand the channel dimension in the output.
        output_c = output_c.axis_to_mul()
        output_c = e3nn.flax.Linear(output_c.irreps, name="linear_out")(output_c)
        return output_c