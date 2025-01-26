import pytest
import functools

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

from freedom.tensor_products import (
    functional,
)


def get_tensor_product(
    tensor_product_type: str,
    lmax: int,
):
    """Get the tensor product function based on the type."""

    if tensor_product_type == "clebsch-gordan-sparse":
        return functional.clebsch_gordan_tensor_product_sparse
    if tensor_product_type == "clebsch-gordan-dense":
        return functional.clebsch_gordan_tensor_product_dense
    if tensor_product_type == "matrix-tp":
        return functional.matrix_tensor_product

    if tensor_product_type == "gaunt-s2grid":
        return functools.partial(
            functional.gaunt_tensor_product_s2grid,
            res_beta=2 * lmax + 1,
            res_alpha=2 * (2 * lmax + 1),
            quadrature="gausslegendre",
            p_val1=1,
            p_val2=1,
            s2grid_fft=False,
        )
    if tensor_product_type == "gaunt-2D-fourier-fft":
        return functools.partial(
            functional.gaunt_tensor_product_2D_fourier,
            res_theta=300,
            res_phi=300,
            convolution_type="fft",
        )
    if tensor_product_type == "gaunt-2D-fourier-direct":
        return functools.partial(
            functional.gaunt_tensor_product_2D_fourier,
            res_theta=300,
            res_phi=300,
            convolution_type="direct",
        )

    if tensor_product_type == "vector-gaunt-s2grid":
        return functools.partial(
            functional.vector_gaunt_tensor_product_s2grid,
            res_beta=2 * lmax + 1,
            res_alpha=2 * (2 * lmax + 1),
            quadrature="gausslegendre",
            p_val1=-1,
            p_val2=-1,
        )

    raise ValueError(f"Unknown tensor product type: {tensor_product_type}")


@pytest.mark.parametrize(
    "lmax",
    [
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "seed",
    [
        0,
        1,
        2,
        3,
        4,
        5,
    ],
)
@pytest.mark.parametrize(
    "tensor_product_type",
    [
        # "clebsch-gordan-dense",
        # "clebsch-gordan-sparse",
        # "gaunt-s2grid",
        "gaunt-2D-fourier-fft",
        "gaunt-2D-fourier-direct",
    ],
)
def test_equivariance(
    lmax: int,
    seed: int,
    tensor_product_type: str,
):
    """Tests for rotational equivariance of tensor products."""

    if tensor_product_type.startswith("gaunt"):
        irreps = e3nn.s2_irreps(lmax)
    else:
        # The factor of 2 is arbitrary.
        irreps = e3nn.s2_irreps(lmax) * 2

    rng = jax.random.PRNGKey(seed)
    (
        R_rng,
        x1_rng,
        x2_rng,
    ) = jax.random.split(
        rng,
        3,
    )
    x1 = e3nn.normal(
        irreps,
        key=x1_rng,
    )
    x2 = e3nn.normal(
        irreps,
        key=x2_rng,
    )

    R = e3nn.rand_matrix(key=R_rng)
    x1_rot = x1.transform_by_matrix(R)
    x2_rot = x2.transform_by_matrix(R)

    tensor_product_fn = get_tensor_product(
        tensor_product_type,
        lmax,
    )
    tp = tensor_product_fn(
        x1,
        x2,
    )
    tp_rot = tensor_product_fn(
        x1_rot,
        x2_rot,
    )

    if tensor_product_type.startswith("gaunt-2D-fourier"):
        atol = 1e-1
    elif tensor_product_type.endswith("s2grid"):
        atol = 1e-2
    else:
        atol = 1e-3

    diff = jnp.abs(tp_rot.array - tp.transform_by_matrix(R).array)
    max_abs_error = diff.max()

    assert jax.numpy.allclose(
        tp_rot.array,
        tp.transform_by_matrix(R).array,
        rtol=0,
        atol=atol,
    ), f"Max absolute error: {max_abs_error:.2e}, "
