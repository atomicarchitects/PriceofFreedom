"""Parameterized tensor products for use in neural networks."""

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

    p_val1: int
    p_val2: int
    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str

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
        output_c = functional.gaunt_tensor_product_s2grid(
            input1_c,
            input2_c,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val1=self.p_val1,
            p_val2=self.p_val2,
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


class VectorGauntTensorProductAllParitiesS2Grid(nn.Module):
    """Vector gaunt tensor product concatenated over all parities."""

    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str
    lmax_grid: bool = True

    @nn.compact
    def __call__(self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        tp1 = VectorGauntTensorProductS2Grid(
            p_val1=1,
            p_val2=1,
            num_channels=self.num_channels,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp2 = VectorGauntTensorProductS2Grid(
            p_val1=1,
            p_val2=-1,
            num_channels=self.num_channels,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp3 = VectorGauntTensorProductS2Grid(
            p_val1=-1,
            p_val2=1,
            num_channels=self.num_channels,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class VectorGauntTensorProductS2Grid(nn.Module):
    """Vector gaunt tensor product using signals on S2."""

    p_val1: int
    p_val2: int
    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str

    @nn.compact
    def __call__(self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # Handle scalar parts separately.
        input1_scalar = input1.filter(lmax=0)
        input1 = input1.filter(drop=["0e", "0o"])
        input1_lmax = input1.irreps.lmax if len(input1.irreps) > 0 else 0

        input1_c = e3nn.flax.Linear(
            VSHCoeffs.get_vsh_irreps(jmax=max(1, input1_lmax - 1), parity=self.p_val1) * self.num_channels,
            force_irreps_out=True,
            name="linear_input1",
        )(input1)
        input1_c = input1_c.mul_to_axis(self.num_channels)

        input2_scalar = input2.filter(lmax=0)
        input2 = input2.filter(drop=["0e", "0o"])
        input2_lmax = input2.irreps.lmax if len(input2.irreps) > 0 else 0

        input2_c = e3nn.flax.Linear(
            VSHCoeffs.get_vsh_irreps(jmax=max(1, input2_lmax - 1), parity=self.p_val2) * self.num_channels,
            force_irreps_out=True,
            name="linear_input2",
        )(input2)
        input2_c = input2_c.mul_to_axis(self.num_channels)

        output_c = functional.vector_gaunt_tensor_product_s2grid(
            input1_c,
            input2_c,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val1=self.p_val1,
            p_val2=self.p_val2,
        )
        output_c = output_c.axis_to_mul()
        output_scalar = functional.clebsch_gordan_tensor_product_dense(input1_scalar, input2_scalar)
        output_c = e3nn.concatenate([output_c, output_scalar])
        output_c = e3nn.flax.Linear(output_c.irreps, name="linear_out")(output_c)
        return output_c
