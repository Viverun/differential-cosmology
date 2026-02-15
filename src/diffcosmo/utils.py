"""Utility functions for differentiable cosmology."""

import jax.numpy as jnp
from typing import Tuple

def compute_k_values(grid_shape: Tuple[int, ...]) -> jnp.ndarray:
    """Compute wavenumber array for FFT.
    
    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the grid (N,) for 1D, (N, N) for 2D, (N, N, N) for 3D
    
    Returns
    -------
    k_values : jnp.ndarray
        Wavenumber magnitudes, same shape as grid_shape
    
    Examples
    --------
    >>> k = compute_k_values((64, 64))
    >>> k.shape
    (64, 64)
    """
    ndim = len(grid_shape)
    k_components = jnp.meshgrid(
        *[jnp.fft.fftfreq(n, d=1.0/n) for n in grid_shape],
        indexing='ij'
    )
    k_squared = sum(k**2 for k in k_components)
    return jnp.sqrt(k_squared)


if __name__ == "__main__":
    # Quick test
    k = compute_k_values((64, 64))
    print(f"k shape: {k.shape}")
    print(f"k range: [{k.min():.2f}, {k.max():.2f}]")
    print("✅ utils.py works!")