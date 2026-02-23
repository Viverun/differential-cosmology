"""Tests for PM evolution."""

import jax
import jax.numpy as jnp

from diffcosmo.fields import generate_gaussian_field, power_law_power_spectrum
from diffcosmo.pm import evolve_pm

COSMOLOGY = {"Omega_m": 0.3, "init_disp_scale": 0.2}


def _power_fn(k):
    return power_law_power_spectrum(k, amplitude=1.0, index=-2.0)


def test_evolve_pm_shape_and_finite():
    field0 = generate_gaussian_field((32, 32), _power_fn, seed=0)
    fieldf = evolve_pm(field0, n_steps=4, dt=0.1, cosmology=COSMOLOGY)

    assert fieldf.shape == field0.shape
    assert jnp.all(jnp.isfinite(fieldf))


def test_evolve_pm_changes_field():
    field0 = generate_gaussian_field((32, 32), _power_fn, seed=1)
    fieldf = evolve_pm(field0, n_steps=4, dt=0.1, cosmology=COSMOLOGY)

    assert not jnp.allclose(field0, fieldf)


def test_evolve_pm_is_differentiable():
    field0 = generate_gaussian_field((16, 16), _power_fn, seed=2)

    def objective(x):
        out = evolve_pm(x, n_steps=2, dt=0.1, cosmology=COSMOLOGY)
        return jnp.mean(out**2)

    grad = jax.grad(objective)(field0)
    assert grad.shape == field0.shape
    assert jnp.all(jnp.isfinite(grad))
