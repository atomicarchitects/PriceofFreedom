from typing import Tuple, Union, Sequence, Optional
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

from src.tensor_products import gaunt_tensor_product_utils as gtp_utils


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


def clebsch_gordan_tensor_product_sparse(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out: Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep], None]] = None,
    irrep_normalization: Optional[str] = None,
) -> e3nn.IrrepsArray:
    """Sparse version of Clebsch-Gordan tensor product."""
    input1, input2, _ = _prepare_inputs(input1, input2)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    if input1.ndim != input2.ndim:
        raise ValueError(f"Inputs must have the same number of dimensions: received {input1.shape} and {input2.shape}")

    tensor_product_fn = lambda x, y: _clebsch_gordan_tensor_product_sparse_single_sample(
        x, y, filter_ir_out=filter_ir_out, irrep_normalization=irrep_normalization
    )
    for _ in range(input1.ndim - 1):
        tensor_product_fn = jax.vmap(tensor_product_fn)
    return tensor_product_fn(input1, input2)


def _clebsch_gordan_tensor_product_sparse_single_sample(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out: Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep], None]] = None,
    irrep_normalization: Optional[str] = None,
) -> e3nn.IrrepsArray:
    """Single-sample version of the sparse version of Clebsch-Gordan tensor product."""
    assert input1.ndim == input2.ndim == 1
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
                            if irrep_normalization == "norm":
                                cg_coeff *= jnp.sqrt(ir_1.dim * ir_2.dim)
                            path *= cg_coeff
                            sum += path
                    chunk = chunk.at[l3 + m3].set(sum)

                chunk = jnp.moveaxis(chunk, 0, -1)
                chunk = jnp.reshape(
                    chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                )
                chunks.append(chunk)

    output = e3nn.from_chunks(irreps_out, chunks, (), input1.dtype)
    output = output.sort()

    return output


def clebsch_gordan_tensor_product_dense(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out=Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep], None]],
    irrep_normalization: Optional[str] = None,
) -> e3nn.IrrepsArray:
    """Dense version of Clebsch-Gordan tensor product."""
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
                    if irrep_normalization == "norm":
                        cg_coeff *= jnp.sqrt(ir_1.dim * ir_2.dim)

                    chunk = jnp.einsum("...ui, ...vj, ijk -> ...uvk", x1, x2, cg_coeff)
                    chunk = jnp.reshape(
                        chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                    )
                else:
                    chunk = None

                chunks.append(chunk)

    output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
    output = output.sort()

    return output


def gaunt_tensor_product_fourier_2D(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    res_theta: int,
    res_phi: int,
    convolution_type: str,
    filter_ir_out=None,
) -> e3nn.IrrepsArray:
    """Gaunt tensor product using 2D Fourier functions."""
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    # Pad the inputs with zeros.
    lmax1 = input1.irreps.lmax
    if input1.irreps != e3nn.s2_irreps(lmax1):
        input1 = input1.extend_with_zeros(e3nn.s2_irreps(lmax1))

    lmax2 = input2.irreps.lmax
    if input2.irreps != e3nn.s2_irreps(lmax2):
        input2 = input2.extend_with_zeros(e3nn.s2_irreps(lmax2))

    with jax.ensure_compile_time_eval():
        # Precompute the change of basis matrices.
        y1_grid = gtp_utils.compute_y_grid(
            lmax1, res_theta=res_theta, res_phi=res_phi)
        y2_grid = gtp_utils.compute_y_grid(
            lmax2, res_theta=res_theta, res_phi=res_phi)
        z_grid = gtp_utils.compute_z_grid(
            lmax1 + lmax2, res_theta=res_theta, res_phi=res_phi
        )

    # Convert to 2D Fourier coefficients.
    input1_uv = jnp.einsum("...a,auv->...uv", input1.array, y1_grid)
    input2_uv = jnp.einsum("...a,auv->...uv", input2.array, y2_grid)

    # Perform the convolution in Fourier space, either directly or using FFT.
    if convolution_type == "direct":
        output_uv = gtp_utils.convolve_2D_direct(input1_uv, input2_uv)
    elif convolution_type == "fft":
        output_uv = gtp_utils.convolve_2D_fft(input1_uv, input2_uv)
    else:
        raise ValueError(f"Unknown convolution type {convolution_type}.")

    # Convert back to SH coefficients.
    output_lm = jnp.einsum("...uv,auv->...a", output_uv.conj(), z_grid)
    output_lm = e3nn.IrrepsArray(
        e3nn.s2_irreps(lmax1 + lmax2),
        output_lm.real,
    )
    return output_lm.filter(filter_ir_out)


def gaunt_tensor_product_s2grid(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    res_beta: int,
    res_alpha: int,
    quadrature: str,
    p_val1: int,
    p_val2: int,
    s2grid_fft: bool,
    filter_ir_out=None,
) -> e3nn.IrrepsArray:
    """Gaunt tensor product using signals on S2."""
    if filter_ir_out is None:
        filter_ir_out = e3nn.s2_irreps(
            input1.irreps.lmax + input2.irreps.lmax, p_val=p_val1 * p_val2, p_arg=-1
        )
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    # Transform the inputs to signals on S2.
    input1_on_grid = e3nn.to_s2grid(
        input1,
        res_beta=res_beta,
        res_alpha=res_alpha,
        quadrature=quadrature,
        p_val=p_val1,
        p_arg=-1,
        fft=False,
    )
    input2_on_grid = e3nn.to_s2grid(
        input2,
        res_beta=res_beta,
        res_alpha=res_alpha,
        quadrature=quadrature,
        p_val=p_val2,
        p_arg=-1,
        fft=False,
    )

    # Multiply the signals on the grid.
    output_on_grid = input1_on_grid * input2_on_grid

    # Transform the output back to irreps.
    output = e3nn.from_s2grid(
        output_on_grid,
        irreps=filter_ir_out,
        fft=s2grid_fft,
    )
    return output
