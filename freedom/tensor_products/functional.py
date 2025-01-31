"""Functional implementations of tensor products."""

from typing import Tuple, Union, Sequence, Optional
import jax
import jax.numpy as jnp
import numpy as np
import math
import e3nn_jax as e3nn
from jax.experimental import sparse

from freedom.tensor_products import gaunt_tensor_product_utils as gtp_utils
from functools import partial


def _prepare_inputs(input1: jnp.ndarray, input2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[int, ...]]:
    """Broadcasts the inputs to a common shape."""
    input1 = e3nn.as_irreps_array(input1)
    input2 = e3nn.as_irreps_array(input2)

    leading_shape = jnp.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape


def _validate_filter_ir_out(
    filter_ir_out: Union[str, e3nn.Irrep, Sequence[e3nn.Irrep]],
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


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def compute_sum(x1_t, x2_t, l1, l2, l3, irrep_normalization):
    with jax.ensure_compile_time_eval():
        cg = e3nn.clebsch_gordan(l1, l2, l3)
        if irrep_normalization == "norm":
            cg *= np.sqrt((2 * l1 + 1) * (2 * l2 + 1))
        A, B, C, D = jnp.zeros((4, 2 * l1 + 1, 2 * l3 + 1))
        for m1 in range(-l1, l1 + 1):
            for m3 in range(-l3, l3 + 1):
                m2 = m1 + m3
                if m2 <= l2 and m2 >= -l2:
                    A = A.at[m1, m3].set(cg[l1 + m1][l2 + m2][l3 + m3])
                    cg[l1 + m1][l2 + m2][l3 + m3] = 0

                m2 = m1 - m3
                if m2 <= l2 and m2 >= -l2:
                    B = B.at[m1, m3].set(cg[l1 + m1][l2 + m2][l3 + m3])
                    cg[l1 + m1][l2 + m2][l3 + m3] = 0

                m2 = -m1 + m3
                if m2 <= l2 and m2 >= -l2:
                    C = C.at[m1, m3].set(cg[l1 + m1][l2 + m2][l3 + m3])
                    cg[l1 + m1][l2 + m2][l3 + m3] = 0

                m2 = -m1 - m3
                if m2 <= l2 and m2 >= -l2:
                    D = D.at[m1, m3].set(cg[l1 + m1][l2 + m2][l3 + m3])
                    cg[l1 + m1][l2 + m2][l3 + m3] = 0

    def m3_func(m3):
        def m1_func(m1):
            m2_A = m1 + m3
            m2_B = m1 - m3
            m2_C = -m1 + m3
            m2_D = -m1 - m3

            valid_A = jnp.logical_and(m2_A >= -l2, m2_A <= l2)
            valid_B = jnp.logical_and(m2_B >= -l2, m2_B <= l2)
            valid_C = jnp.logical_and(m2_C >= -l2, m2_C <= l2)
            valid_D = jnp.logical_and(m2_D >= -l2, m2_D <= l2)

            sum_A = jnp.where(
                valid_A,
                jnp.einsum("u..., v... -> uv...", x1_t[l1 + m1, ...], x2_t[l2 + m2_A, ...]) * A[m1, m3],
                0,
            )
            sum_B = jnp.where(
                valid_B,
                jnp.einsum("u..., v... -> uv...", x1_t[l1 + m1, ...], x2_t[l2 + m2_B, ...]) * B[m1, m3],
                0,
            )
            sum_C = jnp.where(
                valid_C,
                jnp.einsum("u..., v... -> uv...", x1_t[l1 + m1, ...], x2_t[l2 + m2_C, ...]) * C[m1, m3],
                0,
            )
            sum_D = jnp.where(
                valid_D,
                jnp.einsum("u..., v... -> uv...", x1_t[l1 + m1, ...], x2_t[l2 + m2_D, ...]) * D[m1, m3],
                0,
            )

            return sum_A + sum_B + sum_C + sum_D

        return jnp.sum(jax.vmap(m1_func, out_axes=0)(jnp.arange(-l1, l1 + 1)), axis=0)

    return jax.vmap(m3_func, out_axes=0)(jnp.arange(-l3, l3 + 1))


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
                chunk = compute_sum(x1_t, x2_t, l1, l2, l3, irrep_normalization)
                chunk = jnp.moveaxis(chunk, 0, -1)
                chunk = jnp.reshape(chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim))
                chunks.append(chunk)

    output = e3nn.from_chunks(irreps_out, chunks, (), input1.dtype)
    output = output.sort()

    return output


def clebsch_gordan_tensor_product_dense(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out: Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep]]] = None,
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
                    chunk = jnp.reshape(chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim))
                else:
                    chunk = None

                chunks.append(chunk)

    output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
    output = output.sort()

    return output


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def big_CG(
    max_degree1: int,
    max_degree2: int,
    max_degree3: int,
    irrep_normalization: str = "norm",
):
    big_CG = jnp.zeros(((max_degree1 + 1) ** 2, (max_degree2 + 1) ** 2, (max_degree3 + 1) ** 2))
    for l1 in range(max_degree1 + 1):
        for l2 in range(max_degree2 + 1):
            for l3 in range(max_degree3 + 1):
                cg = e3nn.clebsch_gordan(l1, l2, l3)
                if irrep_normalization == "component":
                    cg *= jnp.sqrt(2 * l3 + 1)
                elif irrep_normalization == "norm":
                    cg *= jnp.sqrt((2 * l1 + 1) * (2 * l2 + 1))
                else:
                    raise ValueError(f"Unknown irrep_normalization: {irrep_normalization}")
                big_CG = big_CG.at[l1**2 : (l1 + 1) ** 2, l2**2 : (l2 + 1) ** 2, l3**2 : (l3 + 1) ** 2].set(cg)
    return big_CG


def matrix_tensor_product(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out=None,
    irrep_normalization: Optional[str] = "norm",
):
    """
    Fused Tensor Product from https://e3x.readthedocs.io/stable/_autosummary/e3x.nn.modules.FusedTensor.html#e3x.nn.modules.FusedTensor
    without the weights and ignoring the parity for now. Also both inputs need to have same number of channels
    """
    # TODO: Need a check that explicitly only allows one l or discards them out safely

    irreps_out = []
    chunks = []

    input1, input2, leading_shape = _prepare_inputs(input1, input2)

    with jax.ensure_compile_time_eval():
        # https://github.com/google-research/e3x/blob/ab86199fc3f5fb0663b5798abd62ecf0747a953c/e3x/nn/modules.py#L823
        tilde_l = math.ceil((input1.irreps.lmax + input2.irreps.lmax) / 2)

        # 1. Transform the 2 irreps to matrix irreps space of degree tilde_l
        # 2. Add up all of the matrix irreps (can also add weights here) assuming that all of them have the same multipliticity

        output_ls = []
        for mul_1, ir_1 in input1.irreps:
            for mul_2, ir_2 in input2.irreps:
                for ir_out in ir_1 * ir_2:
                    if (filter_ir_out is not None) and (ir_out not in filter_ir_out) or (ir_out.l in output_ls):
                        continue
                    output_ls.append(ir_out.l)
                    irreps_out.append(ir_out)

        irreps_out = e3nn.Irreps(irreps_out)

        big_C1_tilde = big_CG(tilde_l, tilde_l, input1.irreps.lmax, irrep_normalization)
        big_C2_tilde = big_CG(tilde_l, tilde_l, input2.irreps.lmax, irrep_normalization)
        big_C3_tilde = big_CG(tilde_l, tilde_l, irreps_out.lmax, irrep_normalization)

    # Currently only accepting batch dimension
    X_1 = jnp.einsum("ijC, ...C -> ...ij", big_C1_tilde, input1.array)

    # Assuming that input2 only has one multipliticity
    X_2 = jnp.einsum("jkC, ...C -> ...jk", big_C2_tilde, input2.array)

    # 3. Couple the 2 matrix-reps by matrix-multiplication
    X_3 = jnp.einsum("...ij, ...jk -> ...ik", X_1, X_2)  # Another opportunity to add weights

    # 4. Decompose the matrix reps into output irreps
    output = jnp.einsum("ikD, ...ik -> ...D", big_C3_tilde, X_3)
    # output = output.reshape(leading_shape + (output.shape[-2] * output.shape[-3], irreps_out.dim,))

    output = e3nn.IrrepsArray(irreps_out, output)
    return output


def _get_p_val(irreps: e3nn.Irreps) -> int:
    """Returns the parity value of the irreps."""
    p_val_bools = [(ir.p) ** (ir.l + 1) == 1 for mul, ir in irreps]
    if all(p_val_bools):
        return 1

    if all([not p for p in p_val_bools]):
        return -1

    raise ValueError("Irreps must be either all even or all odd.")


def gaunt_tensor_product_2D_fourier(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    res_theta: int,
    res_phi: int,
    convolution_type: str = "direct",
    filter_ir_out: Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep]]] = None,
    sparsify: bool = True,
) -> e3nn.IrrepsArray:
    """Gaunt tensor product using 2D Fourier functions."""
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    # Pad the inputs with zeros.
    lmax1 = input1.irreps.lmax
    p1 = _get_p_val(input1.irreps)
    if input1.irreps != e3nn.s2_irreps(lmax1, p_val=p1):
        input1 = input1.extend_with_zeros(e3nn.s2_irreps(lmax1, p_val=p1))

    lmax2 = input2.irreps.lmax
    p2 = _get_p_val(input2.irreps)
    if input2.irreps != e3nn.s2_irreps(lmax2, p_val=p2):
        input2 = input2.extend_with_zeros(e3nn.s2_irreps(lmax2, p_val=p2))

    with jax.ensure_compile_time_eval():
        # Precompute the change of basis matrices.
        y1_grid = gtp_utils.compute_y_grid(lmax1, res_theta=res_theta, res_phi=res_phi)
        y2_grid = gtp_utils.compute_y_grid(lmax2, res_theta=res_theta, res_phi=res_phi)
        z_grid = gtp_utils.compute_z_grid(lmax1 + lmax2, res_theta=res_theta, res_phi=res_phi)

        # Convert to sparse arrays.
        if sparsify:
            y1_grid = sparse.BCOO.fromdense(y1_grid.round(8))
            y2_grid = sparse.BCOO.fromdense(y2_grid.round(8))
            z_grid = sparse.BCOO.fromdense(z_grid.round(8))

    @sparse.sparsify
    def to_2D_fourier_coeffs(input: jnp.ndarray, y_grid: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...a,auv->...uv", input, y_grid)

    # Convert to 2D Fourier coefficients.
    input1_uv = to_2D_fourier_coeffs(input1.array, y1_grid)
    input2_uv = to_2D_fourier_coeffs(input2.array, y2_grid)

    # Perform the convolution in Fourier space, either directly or using FFT.
    if convolution_type == "direct":
        output_uv = gtp_utils.convolve_2D_direct(input1_uv, input2_uv)
    elif convolution_type == "fft":
        output_uv = gtp_utils.convolve_2D_fft(input1_uv, input2_uv)
    else:
        raise ValueError(f"Unknown convolution type {convolution_type}.")

    @sparse.sparsify
    def to_SH_coeffs(input: jnp.ndarray, z_grid: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...uv,auv->...a", input.conj(), z_grid)

    # Convert back to SH coefficients.
    output_lm = to_SH_coeffs(output_uv, z_grid)
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
    s2grid_fft: bool = False,
    filter_ir_out=None,
) -> e3nn.IrrepsArray:
    """Gaunt tensor product using signals on S2."""
    if filter_ir_out is None:
        filter_ir_out = e3nn.s2_irreps(input1.irreps.lmax + input2.irreps.lmax, p_val=p_val1 * p_val2, p_arg=-1)
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


def _vector_gaunt_tensor_product_s2grid_single_sample(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    res_beta: int,
    res_alpha: int,
    quadrature: str,
    p_val1: int,
    p_val2: int,
    filter_ir_out: Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep]]] = None,
) -> e3nn.IrrepsArray:
    """Vector Gaunt tensor product using vector signals on S2."""
    input1 = VSHCoeffs(input1, parity=p_val1)
    input2 = VSHCoeffs(input2, parity=p_val2)
    output_on_grid = input1.reduce_pointwise_cross_product(
        input2,
        res_beta=res_beta,
        res_alpha=res_alpha,
        quadrature=quadrature,
    )
    output = output_on_grid.to_irreps_array()
    return output


def vector_gaunt_tensor_product_s2grid(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    res_beta: int,
    res_alpha: int,
    quadrature: str,
    p_val1: int,
    p_val2: int,
    filter_ir_out: Optional[Union[str, e3nn.Irrep, Sequence[e3nn.Irrep]]] = None,
) -> e3nn.IrrepsArray:
    """Vector Gaunt Tensor Product using S2 signals"""
    if filter_ir_out is None:
        filter_ir_out = e3nn.s2_irreps(input1.irreps.lmax + input2.irreps.lmax, p_val=p_val1 * p_val2, p_arg=-1)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    if input1.ndim != input2.ndim:
        raise ValueError(f"Inputs must have the same number of dimensions: received {input1.shape} and {input2.shape}")

    tensor_product_fn = lambda x, y: _vector_gaunt_tensor_product_s2grid_single_sample(
        x,
        y,
        res_beta=res_beta,
        res_alpha=res_alpha,
        quadrature=quadrature,
        p_val1=p_val1,
        p_val2=p_val2,
        filter_ir_out=filter_ir_out,
    )
    for _ in range(input1.ndim - 1):
        tensor_product_fn = jax.vmap(tensor_product_fn)
    return tensor_product_fn(input1, input2)
