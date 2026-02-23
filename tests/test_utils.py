"""Tests for utility functions."""

import jax.numpy as jnp

from diffcosmo.utils import compute_k_values, compute_power_spectrum, cross_correlation


def test_compute_k_values_shape_2d():
    k = compute_k_values((64, 64))
    assert k.shape == (64, 64)


def test_compute_k_values_shape_3d():
    k = compute_k_values((16, 16, 16))
    assert k.shape == (16, 16, 16)


def test_compute_k_values_positive_and_origin_zero():
    k = compute_k_values((64, 64))
    assert jnp.all(k >= 0)
    assert k[0, 0] == 0.0


def test_compute_power_spectrum_returns_nonempty_bins():
    x = jnp.arange(64)
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    field = jnp.sin(2.0 * jnp.pi * xx / 64.0) + jnp.cos(4.0 * jnp.pi * yy / 64.0)
    k, p = compute_power_spectrum(field, n_bins=24)

    assert k.ndim == 1 and p.ndim == 1
    assert k.shape == p.shape
    assert k.size > 0
    assert jnp.all(p >= 0)


def test_cross_correlation_identity_and_orthogonality_trend():
    a = jnp.arange(16 * 16, dtype=jnp.float32).reshape(16, 16)
    b = a.copy()
    c = -a

    assert float(cross_correlation(a, b)) > 0.999
    assert float(cross_correlation(a, c)) < -0.999
