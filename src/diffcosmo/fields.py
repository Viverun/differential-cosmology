"""Initial condition generation."""

import jax.numpy as jnp
from typing import Callable


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
    return jnp.where(k > 0, amplitude * k**index, 0.0)


if __name__ == "__main__":
    from diffcosmo.utils import compute_k_values
    import matplotlib.pyplot as plt

    k = compute_k_values((128, 128))
    P_k = power_law_power_spectrum(k)

    plt.figure(figsize=(8, 6))
    plt.hist(k.flatten(), bins=50, weights=P_k.flatten(), alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title("Power Spectrum")
    plt.savefig("power_spectrum_test.png")
    print("✅ Created power_spectrum_test.png")
