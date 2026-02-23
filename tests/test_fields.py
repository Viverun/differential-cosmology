"""Tests for initial condition generation."""

import jax.numpy as jnp

from diffcosmo.fields import generate_gaussian_field, power_law_power_spectrum
from diffcosmo.utils import compute_power_spectrum


def _power_fn(k):
    return power_law_power_spectrum(k, amplitude=1.0, index=-2.0)


def test_generate_gaussian_field_shape_and_finite():
    field = generate_gaussian_field((64, 64), _power_fn, seed=0)
    assert field.shape == (64, 64)
    assert jnp.all(jnp.isfinite(field))


def test_generate_gaussian_field_is_deterministic_by_seed():
    f1 = generate_gaussian_field((32, 32), _power_fn, seed=7)
    f2 = generate_gaussian_field((32, 32), _power_fn, seed=7)
    f3 = generate_gaussian_field((32, 32), _power_fn, seed=8)

    assert jnp.allclose(f1, f2)
    assert not jnp.allclose(f1, f3)


def test_generate_gaussian_field_has_near_zero_mean():
    field = generate_gaussian_field((64, 64), _power_fn, seed=3)
    assert abs(float(jnp.mean(field))) < 1e-4


def test_generate_gaussian_field_has_more_large_scale_power_for_negative_index():
    field = generate_gaussian_field((64, 64), _power_fn, seed=1)
    k, p = compute_power_spectrum(field, n_bins=20)

    # For P(k) ~ k^-2, low-k bins should dominate high-k bins on average.
    assert float(p[1]) > float(p[-1])
