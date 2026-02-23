"""Tests for MAP loss construction."""

import jax
import jax.numpy as jnp

from diffcosmo.fields import generate_gaussian_field, power_law_power_spectrum
from diffcosmo.loss import total_loss
from diffcosmo.pm import evolve_pm

SIM_CONFIG = {
    "n_steps": 4,
    "dt": 0.1,
    "checkpointing": "none",
    "cosmology": {"Omega_m": 0.3, "init_disp_scale": 0.2},
}
PRIOR_CONFIG = {
    "amplitude": 1.0,
    "index": -2.0,
    "lambda_prior": 1.0e-2,
    "eps": 1.0e-6,
}
OBS_CONFIG = {"noise_std": 0.1, "mask": None}


def _power_fn(k):
    return power_law_power_spectrum(k, amplitude=1.0, index=-2.0)


def test_total_loss_scalar_and_finite():
    theta = generate_gaussian_field((16, 16), _power_fn, seed=0)
    y_obs = evolve_pm(theta, n_steps=4, dt=0.1, cosmology=SIM_CONFIG["cosmology"])

    loss = total_loss(theta, y_obs, SIM_CONFIG, PRIOR_CONFIG, OBS_CONFIG)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_total_loss_has_finite_gradients():
    theta = generate_gaussian_field((16, 16), _power_fn, seed=1)
    y_obs = evolve_pm(theta, n_steps=4, dt=0.1, cosmology=SIM_CONFIG["cosmology"])

    grad = jax.grad(total_loss)(theta, y_obs, SIM_CONFIG, PRIOR_CONFIG, OBS_CONFIG)
    assert grad.shape == theta.shape
    assert jnp.all(jnp.isfinite(grad))


def test_total_loss_prefers_better_prediction_when_prior_disabled():
    theta_true = generate_gaussian_field((16, 16), _power_fn, seed=2)
    y_obs = evolve_pm(theta_true, n_steps=4, dt=0.1, cosmology=SIM_CONFIG["cosmology"])
    theta_bad = jnp.zeros_like(theta_true)

    prior_off = dict(PRIOR_CONFIG)
    prior_off["lambda_prior"] = 0.0

    loss_true = total_loss(theta_true, y_obs, SIM_CONFIG, prior_off, OBS_CONFIG)
    loss_bad = total_loss(theta_bad, y_obs, SIM_CONFIG, prior_off, OBS_CONFIG)

    assert float(loss_true) < float(loss_bad)
