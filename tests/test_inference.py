"""Tests for MAP inference."""

import jax.numpy as jnp

from diffcosmo.fields import generate_gaussian_field, power_law_power_spectrum
from diffcosmo.inference import reconstruct_map
from diffcosmo.observe import apply_observation_model
from diffcosmo.pm import evolve_pm
from diffcosmo.utils import cross_correlation

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
OPT_CONFIG = {
    "lr": 0.05,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1.0e-8,
    "max_iters": 60,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1.0e-6,
}


def _power_fn(k):
    return power_law_power_spectrum(k, amplitude=1.0, index=-2.0)


def test_reconstruct_map_decreases_loss_and_improves_correlation():
    grid_shape = (16, 16)
    theta_true = generate_gaussian_field(grid_shape, _power_fn, seed=0)
    delta_true = evolve_pm(theta_true, n_steps=4, dt=0.1, cosmology=SIM_CONFIG["cosmology"])
    y_obs = apply_observation_model(delta_true, noise_std=0.05, mask=None, seed=0)

    theta_init = generate_gaussian_field(grid_shape, _power_fn, seed=1)
    obs_config = {"noise_std": 0.05, "mask": None}

    init_corr = float(cross_correlation(theta_true, theta_init))
    result = reconstruct_map(y_obs, theta_init, SIM_CONFIG, PRIOR_CONFIG, obs_config, OPT_CONFIG)

    loss_history = result["loss_history"]
    theta_map = result["theta_map"]
    final_corr = float(cross_correlation(theta_true, theta_map))

    assert loss_history.size > 2
    assert float(loss_history[-1]) < float(loss_history[0])
    assert final_corr > init_corr
    assert jnp.all(jnp.isfinite(theta_map))
