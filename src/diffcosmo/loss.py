"""Loss functions for MAP reconstruction."""

from typing import Any, Dict

import jax.numpy as jnp

from diffcosmo.fields import power_law_power_spectrum
from diffcosmo.pm import evolve_pm
from diffcosmo.utils import compute_k_values


def data_misfit(y_pred: jnp.ndarray, y_obs: jnp.ndarray, noise_std: float) -> jnp.ndarray:
    """Gaussian negative log-likelihood up to constants."""
    residual = y_pred - y_obs
    return 0.5 * jnp.mean((residual / noise_std) ** 2)


def prior_penalty(
    theta0: jnp.ndarray,
    amplitude: float = 1.0,
    index: float = -2.0,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Spectral Gaussian prior term sum |theta_k|^2 / P(k)."""
    k_mag = compute_k_values(theta0.shape)
    p_k = power_law_power_spectrum(k_mag, amplitude=amplitude, index=index)

    theta_k = jnp.fft.fftn(theta0)
    spectral_energy = (jnp.abs(theta_k) ** 2) / theta0.size
    weighted = spectral_energy / (p_k + eps)
    return jnp.mean(weighted.real)


def total_loss(
    theta0: jnp.ndarray,
    y_obs: jnp.ndarray,
    sim_config: Dict[str, Any],
    prior_config: Dict[str, Any],
    obs_config: Dict[str, Any],
) -> jnp.ndarray:
    """Compute full MAP objective for initial field reconstruction."""
    y_pred = evolve_pm(
        initial_field=theta0,
        n_steps=int(sim_config["n_steps"]),
        dt=float(sim_config["dt"]),
        cosmology=sim_config["cosmology"],
        checkpointing=sim_config.get("checkpointing", "none"),
    )

    mask = obs_config.get("mask")
    if mask is not None:
        y_pred = y_pred * mask.astype(y_pred.dtype)

    data_term = data_misfit(y_pred, y_obs, noise_std=float(obs_config["noise_std"]))
    prior_term = prior_penalty(
        theta0,
        amplitude=float(prior_config.get("amplitude", 1.0)),
        index=float(prior_config.get("index", -2.0)),
        eps=float(prior_config.get("eps", 1e-6)),
    )
    return data_term + float(prior_config.get("lambda_prior", 1e-2)) * prior_term
