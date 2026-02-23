"""Initial condition generation."""

from typing import Callable, Tuple

import jax.numpy as jnp
import jax.random as random

from diffcosmo.utils import compute_k_values


def power_law_power_spectrum(
    k: jnp.ndarray, amplitude: float = 1.0, index: float = -2.0
) -> jnp.ndarray:
    """Simple power-law power spectrum.

    P(k) = amplitude * k^index

    Parameters
    ----------
    k : jnp.ndarray
        Wavenumber values
    amplitude : float
        Normalization
    index : float
        Power-law index (typically -2 to -3)

    Returns
    -------
    P_k : jnp.ndarray
        Power spectrum values
    """
    k_safe = jnp.where(k > 0, k, 1.0)
    power = amplitude * k_safe**index
    return jnp.where(k > 0, power, 0.0)


def generate_gaussian_field(
    grid_shape: Tuple[int, int],
    power_spectrum_fn: Callable[[jnp.ndarray], jnp.ndarray],
    seed: int,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Generate a real Gaussian random field with target power spectrum.

    The method samples white noise in real space and filters in Fourier space
    with sqrt(P(k)); inverse FFT yields a real field.
    """
    if len(grid_shape) != 2:
        raise ValueError("generate_gaussian_field currently supports 2D grids only.")

    key = random.PRNGKey(seed)
    white_noise = random.normal(key, shape=grid_shape, dtype=dtype)
    white_k = jnp.fft.fftn(white_noise)

    k_mag = compute_k_values(grid_shape)
    power = jnp.maximum(power_spectrum_fn(k_mag), 0.0)
    filter_amp = jnp.sqrt(power + 1e-8)

    colored_k = white_k * filter_amp
    field = jnp.real(jnp.fft.ifftn(colored_k)).astype(dtype)
    field = field - jnp.mean(field)
    field = field / (jnp.std(field) + 1e-6)
    return field
