"""Observation operators for toy differentiable cosmology experiments."""

from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as random


def make_rectangular_mask(grid_shape: Tuple[int, int], frac: float = 0.8) -> jnp.ndarray:
    """Create a centered rectangular survey mask with area fraction ~= frac^2."""
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")

    nx, ny = grid_shape
    wx = int(jnp.floor(nx * frac))
    wy = int(jnp.floor(ny * frac))

    x0 = (nx - wx) // 2
    y0 = (ny - wy) // 2

    mask = jnp.zeros((nx, ny), dtype=jnp.float32)
    mask = mask.at[x0 : x0 + wx, y0 : y0 + wy].set(1.0)
    return mask


def apply_observation_model(
    field: jnp.ndarray,
    noise_std: float,
    mask: Optional[jnp.ndarray] = None,
    seed: int = 0,
) -> jnp.ndarray:
    """Apply a survey mask and Gaussian observation noise."""
    if mask is None:
        mask = jnp.ones_like(field, dtype=field.dtype)
    else:
        mask = mask.astype(field.dtype)

    key = random.PRNGKey(seed)
    noise = random.normal(key, shape=field.shape, dtype=field.dtype) * noise_std
    return mask * field + noise
