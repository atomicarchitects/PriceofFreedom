from typing import Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax.linen as nn

from src.tensor_products.vector_spherical_harmonics import VSHCoeffs
from src.tensor_products import functional



class ClebschGordanTensorProductDense(nn.Module):
    """Dense version of Clebsch-Gordan tensor product."""

    irrep_normalization: str
    output_linear: bool

    @nn.compact
    def __call__(
        self,
        input1: e3nn.IrrepsArray,
        input2: e3nn.IrrepsArray,
        *,
        filter_ir_out=None,
    ) -> e3nn.IrrepsArray:
        output = functional.clebsch_gordan_tensor_product_dense(
            input1, input2, filter_ir_out=filter_ir_out, irrep_normalization=self.irrep_normalization
        )
        if self.output_linear:
            output = e3nn.flax.Linear(output.irreps)(output)
        return output


class ClebschGordanTensorProductSparse(nn.Module):

    irrep_normalization: str
    output_linear: bool

    def __call__(
        self,
        input1: e3nn.IrrepsArray,
        input2: e3nn.IrrepsArray,
        *,
        filter_ir_out=None,
    ) -> e3nn.IrrepsArray:
        output = functional.clebsch_gordan_tensor_product_sparse(
            input1, input2, filter_ir_out=filter_ir_out, irrep_normalization=self.irrep_normalization
        )
        if self.output_linear:
            output = e3nn.flax.Linear(output.irreps)(output)

        return output


class GauntTensorProductAllParitiesS2Grid(nn.Module):
    """Gaunt tensor product concatenated over all parities, using signals on S2."""

    lmax: int
    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        tp1 = GauntTensorProductS2Grid(
            lmax=self.lmax,
            num_channels=self.num_channels,
            p_val1=1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp2 = GauntTensorProductS2Grid(
            lmax=self.lmax,
            num_channels=self.num_channels,
            p_val1=1,
            p_val2=-1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp3 = GauntTensorProductS2Grid(
            lmax=self.lmax,
            num_channels=self.num_channels,
            p_val1=-1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class GauntTensorProductS2Grid(nn.Module):
    """Gaunt tensor product using signals on S2."""

    lmax: int
    num_channels: int
    p_val1: int
    p_val2: int
    res_alpha: int
    res_beta: int
    quadrature: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        # Project the inputs to the desired parity and channels.
        input1c = e3nn.flax.Linear(
            e3nn.s2_irreps(self.lmax, p_val=self.p_val1, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in1",
        )(input1)
        input1c = input1c.mul_to_axis(self.num_channels)

        input2c = e3nn.flax.Linear(
            e3nn.s2_irreps(self.lmax, p_val=self.p_val2, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in2",
        )(input2)
        input2c = input2c.mul_to_axis(self.num_channels)

        # Compute the tensor product.
        outputc = functional.gaunt_tensor_product_s2grid(
            input1c,
            input2c,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val1=self.p_val1,
            p_val2=self.p_val2,
            s2grid_fft=False,
        )

        # Expand the channel dimension in the output.
        outputc = outputc.axis_to_mul()
        outputc = e3nn.flax.Linear(outputc.irreps, name="linear_out")(outputc)
        return outputc


class GauntTensorProduct2DFourier(nn.Module):
    """Gaunt tensor product using 2D Fourier functions."""

    lmax: int
    num_channels: int
    p_val1: int
    p_val2: int
    res_theta: int
    res_phi: int
    convolution_type: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        # Project the inputs to the desired parity and channels.
        input1c = e3nn.flax.Linear(
            e3nn.s2_irreps(self.lmax, p_val=self.p_val1, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in1",
        )(input1)
        input1c = input1c.mul_to_axis(self.num_channels)

        input2c = e3nn.flax.Linear(
            e3nn.s2_irreps(self.lmax, p_val=self.p_val2, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in2",
        )(input2)
        input2c = input2c.mul_to_axis(self.num_channels)

        # Compute the tensor product.
        outputc = functional.gaunt_tensor_product_fourier_2D(
            input1c,
            input2c,
            res_theta=self.res_theta,
            res_phi=self.res_phi,
            convolution_type=self.convolution_type,
        )

        # Expand the channel dimension in the output.
        outputc = outputc.axis_to_mul()
        outputc = e3nn.flax.Linear(outputc.irreps, name="linear_out")(outputc)
        return outputc


class VectorGauntTensorProductAllParitiesS2Grid(nn.Module):
    """Vector gaunt tensor product concatenated over all parities."""

    lmax: int
    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        tp1 = VectorGauntTensorProductS2Grid(
            lmax=self.lmax,
            p_val1=1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp2 = VectorGauntTensorProductS2Grid(
            lmax=self.lmax,
            p_val1=1,
            p_val2=-1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp3 = VectorGauntTensorProductS2Grid(
            lmax=self.lmax,
            p_val1=-1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class VectorGauntTensorProductS2Grid(nn.Module):
    """Vector gaunt tensor product using signals on S2."""

    lmax: int
    p_val1: int
    p_val2: int
    res_alpha: int
    res_beta: int
    num_channels: int
    quadrature: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        input1c = e3nn.flax.Linear(
            VSHCoeffs.get_vsh_irreps(self.lmax, parity=self.p_val1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_input1",
        )(input1)
        input1c = input1c.mul_to_axis(self.num_channels)

        input2c = e3nn.flax.Linear(
            VSHCoeffs.get_vsh_irreps(self.lmax, parity=self.p_val2)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_input2",
        )(input2)
        input2c = input2c.mul_to_axis(self.num_channels)

        def cross_product_per_channel_per_sample(input1c, input2c):
            input1c = VSHCoeffs(input1c, parity=self.p_val1)
            input2c = VSHCoeffs(input2c, parity=self.p_val2)
            output_on_grid = input1c.reduce_pointwise_cross_product(
                input2c,
                res_beta=self.res_beta,
                res_alpha=self.res_alpha,
                quadrature=self.quadrature,
            )
            outputc = output_on_grid.to_irreps_array()
            return outputc

        outputc = jax.vmap(jax.vmap(cross_product_per_channel_per_sample))(
            input1c, input2c
        )
        outputc = outputc.axis_to_mul()
        outputc = e3nn.flax.Linear(outputc.irreps, name="linear_out_z")(outputc)
        return outputc
