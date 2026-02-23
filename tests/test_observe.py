"""Tests for observation operators."""

import jax.numpy as jnp

from diffcosmo.observe import apply_observation_model, make_rectangular_mask


def test_make_rectangular_mask_shape_and_values():
    mask = make_rectangular_mask((32, 32), frac=0.5)
    assert mask.shape == (32, 32)
    assert jnp.min(mask) >= 0.0
    assert jnp.max(mask) <= 1.0


def test_apply_observation_model_is_deterministic_for_seed():
    field = jnp.ones((16, 16), dtype=jnp.float32)
    y1 = apply_observation_model(field, noise_std=0.1, seed=4)
    y2 = apply_observation_model(field, noise_std=0.1, seed=4)
    y3 = apply_observation_model(field, noise_std=0.1, seed=5)

    assert jnp.allclose(y1, y2)
    assert not jnp.allclose(y1, y3)


def test_apply_observation_model_respects_mask_when_noise_is_zero():
    field = jnp.arange(16 * 16, dtype=jnp.float32).reshape(16, 16)
    mask = make_rectangular_mask((16, 16), frac=0.5)
    y = apply_observation_model(field, noise_std=0.0, mask=mask, seed=0)

    assert jnp.allclose(y, field * mask)
