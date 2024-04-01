"""Tests for vector_spherical_harmonics.py."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import e3nn_jax as e3nn

import sys

sys.path.append(".")
from vector_spherical_harmonics import VSHCoeffs


@pytest.mark.parametrize("key", [0, 1, 2, 3, 4])
def test_to_vector_coeffs_equivariance(key):
    key = jax.random.PRNGKey(key)

    coeffs_key, R_key = jax.random.split(key)
    coeffs = VSHCoeffs.normal(jmax=4, parity=-1, key=coeffs_key)
    irreps = coeffs.to_vector_coeffs()

    alpha, beta, gamma = jax.random.uniform(R_key, (3,))
    R = e3nn.angles_to_matrix(alpha, beta, gamma)
    rotated_coeffs = jax.tree_map(
        lambda x: x.transform_by_matrix(R),
        coeffs,
        is_leaf=lambda x: isinstance(x, e3nn.IrrepsArray),
    )
    rotated_irreps = rotated_coeffs.to_vector_coeffs()

    assert jnp.allclose(
        R @ irreps.transform_by_matrix(R).array, rotated_irreps.array, atol=1e-4
    )


@pytest.mark.parametrize("key", [0, 1, 2, 3, 4])
def test_cross_product_selection_rules(key):
    key = jax.random.PRNGKey(key)

    j1_key, l1_key, coeffs1_key, key = jax.random.split(key, 4)
    j1 = int(jax.random.randint(j1_key, (), 1, 11))
    l1 = int(jax.random.randint(l1_key, (), j1 - 1, j1 + 2))
    coeffs1 = VSHCoeffs(parity=-1)
    coeffs1[(j1, l1)] = e3nn.normal(e3nn.Irrep(j1, (-1) ** (l1 + 1)), coeffs1_key)

    j2_key, l2_key, coeffs2_key, key = jax.random.split(key, 4)
    j2 = int(jax.random.randint(j2_key, (), 1, 11))
    l2 = int(jax.random.randint(l2_key, (), j2 - 1, j2 + 2))
    coeffs2 = VSHCoeffs(parity=-1)
    coeffs2[(j2, l2)] = e3nn.normal(e3nn.Irrep(j2, (-1) ** (l2 + 1)), coeffs2_key)

    cross_product = coeffs1.pointwise_cross_product(
        coeffs2, res_beta=80, res_alpha=49, quadrature="gausslegendre"
    )

    for (j3, l3), coeffs in cross_product.items():
        if jnp.allclose(cross_product[(j3, l3)].array, 0, atol=1e-3):
            continue

        # Check first conditions
        assert l1 - 1 <= j1 <= l1 + 1
        assert l2 - 1 <= j2 <= l2 + 1
        assert l3 - 1 <= j3 <= l3 + 1

        # Check second condition
        assert abs(l1 - l2) <= l3 <= l1 + l2, (l1, l2, l3)

        # Check third condition
        assert abs(j1 - j2) <= j3 <= j1 + j2, (j1, j2, j3)

        # Check fourth condition
        assert (l1 + l2 + l3) % 2 == 0, (l1, l2, l3, cross_product[(j3, l3)])

        # Check fifth condition
        ls = [l1, l2, l3]
        js = [j1, j2, j3]
        for a, b, c in itertools.permutations(range(3)):
            assert (ls[a], ls[b], js[b]) != (js[a], ls[c], js[c])
