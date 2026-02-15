"""Tests for utility functions."""

import pytest
import jax.numpy as jnp
from diffcosmo.utils import compute_k_values


def test_compute_k_values_shape_2d():
    """k should have same shape as input grid."""
    k = compute_k_values((64, 64))
    assert k.shape == (64, 64)


def test_compute_k_values_shape_3d():
    """k should work for 3D grids."""
    k = compute_k_values((32, 32, 32))
    assert k.shape == (32, 32, 32)


def test_compute_k_values_positive():
    """k values should be non-negative."""
    k = compute_k_values((64, 64))
    assert jnp.all(k >= 0)


def test_compute_k_values_origin_zero():
    """k should be zero at origin."""
    k = compute_k_values((64, 64))
    assert k[0, 0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])