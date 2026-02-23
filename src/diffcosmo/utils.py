"""Utility functions for differentiable cosmology."""

from typing import Tuple

import jax.numpy as jnp


def compute_k_values(grid_shape: Tuple[int, ...]) -> jnp.ndarray:
    """Compute wavenumber magnitudes for an FFT grid."""
    k_components = jnp.meshgrid(
        *[jnp.fft.fftfreq(n, d=1.0 / n) for n in grid_shape],
        indexing="ij",
    )
    k_squared = sum(k_comp**2 for k_comp in k_components)
    return jnp.sqrt(k_squared)


def compute_power_spectrum(field: jnp.ndarray, n_bins: int = 32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate isotropic power spectrum via radial binning in k-space."""
    if field.ndim not in (2, 3):
        raise ValueError("compute_power_spectrum supports 2D or 3D fields only.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    k_mag = compute_k_values(field.shape).reshape(-1)
    field_k = jnp.fft.fftn(field)
    power_k = (jnp.abs(field_k) ** 2 / field.size).reshape(-1)

    bins = jnp.linspace(0.0, float(k_mag.max()), n_bins + 1)
    bin_ids = jnp.clip(jnp.digitize(k_mag, bins) - 1, 0, n_bins - 1)

    power_sum = jnp.bincount(bin_ids, weights=power_k, length=n_bins)
    counts = jnp.bincount(bin_ids, length=n_bins)
    power = jnp.where(counts > 0, power_sum / counts, 0.0)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    valid = counts > 0
    return k_centers[valid], power[valid]


def cross_correlation(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Return normalized cross-correlation coefficient between two fields."""
    a_flat = jnp.ravel(a)
    b_flat = jnp.ravel(b)

    a_centered = a_flat - jnp.mean(a_flat)
    b_centered = b_flat - jnp.mean(b_flat)

    numerator = jnp.sum(a_centered * b_centered)
    denominator = jnp.sqrt(jnp.sum(a_centered**2) * jnp.sum(b_centered**2))
    return numerator / (denominator + eps)
