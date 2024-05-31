from typing import Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax.linen as nn

from src.vector_spherical_harmonics import VSHCoeffs


def _prepare_inputs(
    input1: jnp.ndarray, input2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[int, ...]]:
    """Broadcasts the inputs to a common shape."""
    input1 = e3nn.as_irreps_array(input1)
    input2 = e3nn.as_irreps_array(input2)

    leading_shape = jnp.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape


def _validate_filter_ir_out(
    filter_ir_out: Union[str, e3nn.Irrep, Sequence[e3nn.Irrep], None]
):
    """Validates the filter_ir_out argument."""
    if filter_ir_out is not None:
        if isinstance(filter_ir_out, str):
            filter_ir_out = e3nn.Irreps(filter_ir_out)
        if isinstance(filter_ir_out, e3nn.Irrep):
            filter_ir_out = [filter_ir_out]
        filter_ir_out = [e3nn.Irrep(ir) for ir in filter_ir_out]
    return filter_ir_out


class TensorProductNaive(nn.Module):

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
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        filter_ir_out = _validate_filter_ir_out(filter_ir_out)

        irreps_out = []
        chunks = []
        for (mul_1, ir_1), x1 in zip(input1.irreps, input1.chunks):
            for (mul_2, ir_2), x2 in zip(input2.irreps, input2.chunks):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue

                    irreps_out.append((mul_1 * mul_2, ir_out))

                    if x1 is not None and x2 is not None:
                        cg_coeff = e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)
                        cg_coeff = cg_coeff.astype(x1.dtype)
                        if self.irrep_normalization == "norm":
                            cg_coeff *= jnp.sqrt(ir_1.dim * ir_2.dim)

                        chunk = jnp.einsum(
                            "...ui, ...vj, ijk -> ...uvk", x1, x2, cg_coeff
                        )
                        chunk = jnp.reshape(
                            chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                        )
                    else:
                        chunk = None

                    chunks.append(chunk)

        output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
        output = output.sort()

        if self.output_linear:
            output = e3nn.flax.Linear(output.irreps)(output)

        return output


class TensorProductOptimized(nn.Module):

    irrep_normalization: str
    output_linear: bool

    def __call__(
        self,
        input1: e3nn.IrrepsArray,
        input2: e3nn.IrrepsArray,
        *,
        filter_ir_out=None,
    ) -> e3nn.IrrepsArray:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        filter_ir_out = _validate_filter_ir_out(filter_ir_out)

        irreps_out = []
        chunks = []
        for (mul_1, ir_1), x1 in zip(input1.irreps, input1.chunks):
            for (mul_2, ir_2), x2 in zip(input2.irreps, input2.chunks):
                if x1 is None or x2 is None:
                    continue

                x1_t = jnp.moveaxis(x1, -1, 0)
                x2_t = jnp.moveaxis(x2, -1, 0)

                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue

                    irreps_out.append((mul_1 * mul_2, ir_out))

                    l1, l2, l3 = ir_1.l, ir_2.l, ir_out.l
                    cg = e3nn.clebsch_gordan(l1, l2, l3)
                    chunk = jnp.zeros((2 * l3 + 1, x1.shape[-2], x2.shape[-2]))
                    for m3 in range(-l3, l3 + 1):
                        sum = 0
                        for m1 in range(-l1, l1 + 1):
                            for m2 in set([m3 - m1, m3 + m1, -m3 + m1, -m3 - m1]):
                                if m2 < -l2 or m2 > l2:
                                    continue

                                path = jnp.einsum(
                                    "u...,v... -> uv...",
                                    x1_t[l1 + m1, ...],
                                    x2_t[l2 + m2, ...],
                                )
                                cg_coeff = cg[l1 + m1, l2 + m2, l3 + m3]
                                if self.irrep_normalization == "norm":
                                    cg_coeff *= jnp.sqrt(ir_1.dim * ir_2.dim)
                                path *= cg_coeff
                                sum += path
                        chunk = chunk.at[l3 + m3].set(sum)

                    chunk = jnp.moveaxis(chunk, 0, -1)
                    chunk = jnp.reshape(
                        chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                    )
                    chunks.append(chunk)

        output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
        output = output.sort()

        if self.output_linear:
            output = e3nn.flax.Linear(output.irreps)(output)

        return output


class GauntTensorProduct(nn.Module):

    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        tp1 = GauntTensorProductFixedParity(
            p_val1=1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp2 = GauntTensorProductFixedParity(
            p_val1=1,
            p_val2=-1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp3 = GauntTensorProductFixedParity(
            p_val1=-1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class GauntTensorProductFixedParity(nn.Module):

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
        input1c = e3nn.flax.Linear(
            e3nn.s2_irreps(input1.irreps.lmax, p_val=self.p_val1, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in_x",
        )(input1)
        input1c = input1c.mul_to_axis(self.num_channels)

        input2c = e3nn.flax.Linear(
            e3nn.s2_irreps(input2.irreps.lmax, p_val=self.p_val2, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in_y",
        )(input2)
        input2c = input2c.mul_to_axis(self.num_channels)

        input1_on_grid = e3nn.to_s2grid(
            input1c,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val=self.p_val1,
            p_arg=-1,
            fft=False,
        )
        input2_on_grid = e3nn.to_s2grid(
            input2c,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val=self.p_val2,
            p_arg=-1,
            fft=False,
        )
        output_on_grid = input1_on_grid * input2_on_grid
        outputc = e3nn.from_s2grid(
            output_on_grid,
            irreps=e3nn.s2_irreps(
                input1.irreps.lmax + input2.irreps.lmax, p_val=self.p_val1 * self.p_val2
            ),
            fft=False,
        )
        outputc = outputc.axis_to_mul()
        outputc = e3nn.flax.Linear(outputc.irreps, name="linear_out_z")(outputc)
        return outputc


class VectorGauntTensorProduct(nn.Module):

    num_channels: int
    res_alpha: int
    res_beta: int
    quadrature: str

    @nn.compact
    def __call__(
        self, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        tp1 = VectorGauntTensorProductFixedParity(
            p_val1=1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp2 = VectorGauntTensorProductFixedParity(
            p_val1=1,
            p_val2=-1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        tp3 = VectorGauntTensorProductFixedParity(
            p_val1=-1,
            p_val2=1,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
        )(input1, input2)
        return e3nn.concatenate([tp1, tp2, tp3])


class VectorGauntTensorProductFixedParity(nn.Module):

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
            VSHCoeffs.get_vsh_irreps(input1.irreps.lmax, parity=self.p_val1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_input1",
        )(input1)
        input1c = input1c.mul_to_axis(self.num_channels)

        input2c = e3nn.flax.Linear(
            VSHCoeffs.get_vsh_irreps(input2.irreps.lmax, parity=self.p_val2)
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
