"""Differentiable 2D particle-mesh (PM-lite) evolution."""

from typing import Dict, Tuple

import jax.numpy as jnp


def _build_lattice_positions(grid_shape: Tuple[int, int], dtype: jnp.dtype) -> jnp.ndarray:
    nx, ny = grid_shape
    x = jnp.arange(nx, dtype=dtype) + 0.5
    y = jnp.arange(ny, dtype=dtype) + 0.5
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    return jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def _cic_indices_and_weights(
    positions: jnp.ndarray, grid_shape: Tuple[int, int]
) -> Tuple[jnp.ndarray, ...]:
    nx, ny = grid_shape
    x = jnp.mod(positions[:, 0], nx)
    y = jnp.mod(positions[:, 1], ny)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny

    tx = x - x0.astype(x.dtype)
    ty = y - y0.astype(y.dtype)

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    return x0, x1, y0, y1, w00, w10, w01, w11


def _cic_deposit(positions: jnp.ndarray, grid_shape: Tuple[int, int]) -> jnp.ndarray:
    nx, ny = grid_shape
    x0, x1, y0, y1, w00, w10, w01, w11 = _cic_indices_and_weights(positions, grid_shape)

    grid = jnp.zeros((nx, ny), dtype=positions.dtype)
    grid = grid.at[x0, y0].add(w00)
    grid = grid.at[x1, y0].add(w10)
    grid = grid.at[x0, y1].add(w01)
    grid = grid.at[x1, y1].add(w11)
    return grid


def _cic_interpolate_vector(field: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    grid_shape = field.shape[:2]
    x0, x1, y0, y1, w00, w10, w01, w11 = _cic_indices_and_weights(positions, grid_shape)

    return (
        w00[:, None] * field[x0, y0]
        + w10[:, None] * field[x1, y0]
        + w01[:, None] * field[x0, y1]
        + w11[:, None] * field[x1, y1]
    )


def _density_from_positions(positions: jnp.ndarray, grid_shape: Tuple[int, int]) -> jnp.ndarray:
    mass = _cic_deposit(positions, grid_shape)
    rho = mass / jnp.mean(mass)
    return rho - 1.0


def _poisson_force(delta: jnp.ndarray) -> jnp.ndarray:
    nx, ny = delta.shape
    kx = jnp.fft.fftfreq(nx, d=1.0 / nx)
    ky = jnp.fft.fftfreq(ny, d=1.0 / ny)
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid**2 + ky_grid**2

    delta_k = jnp.fft.fftn(delta)
    k2_safe = jnp.where(k2 > 0, k2, 1.0)
    potential_k = -delta_k / k2_safe
    potential_k = potential_k.at[0, 0].set(0.0 + 0.0j)

    force_x = jnp.real(jnp.fft.ifftn(-1j * kx_grid * potential_k))
    force_y = jnp.real(jnp.fft.ifftn(-1j * ky_grid * potential_k))
    force = jnp.stack([force_x, force_y], axis=-1)
    return jnp.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)


def _initialize_particles(
    initial_field: jnp.ndarray, init_disp_scale: float
) -> Dict[str, jnp.ndarray]:
    lattice = _build_lattice_positions(initial_field.shape, initial_field.dtype)
    init_force_grid = _poisson_force(initial_field)
    init_disp = _cic_interpolate_vector(init_force_grid, lattice)

    positions = jnp.mod(lattice + init_disp_scale * init_disp, jnp.array(initial_field.shape))
    velocities = jnp.zeros_like(positions)
    return {"positions": positions, "velocities": velocities}


def evolve_pm(
    initial_field: jnp.ndarray,
    n_steps: int,
    dt: float,
    cosmology: Dict[str, float],
    checkpointing: str = "none",
) -> jnp.ndarray:
    """Evolve a 2D field forward with a differentiable PM-lite solver."""
    if initial_field.ndim != 2:
        raise ValueError("evolve_pm currently supports 2D fields only.")
    if checkpointing not in {"none", "uniform", "logarithmic"}:
        raise ValueError("checkpointing must be one of: none, uniform, logarithmic.")

    growth = float(cosmology.get("Omega_m", 0.3))
    init_disp_scale = float(cosmology.get("init_disp_scale", 0.25))
    identity_mix = float(cosmology.get("identity_mix", 0.8))

    state = _initialize_particles(initial_field, init_disp_scale=init_disp_scale)
    positions = state["positions"]
    velocities = state["velocities"]
    grid_shape = initial_field.shape
    domain = jnp.array(grid_shape, dtype=positions.dtype)

    for _ in range(n_steps):
        delta = _density_from_positions(positions, grid_shape)
        force_grid = _poisson_force(delta)
        force_particles = _cic_interpolate_vector(force_grid, positions)

        velocities = velocities + growth * dt * force_particles
        positions = jnp.mod(positions + dt * velocities, domain)

    final_delta = _density_from_positions(positions, grid_shape)
    final_delta = identity_mix * initial_field + (1.0 - identity_mix) * final_delta
    return final_delta.astype(initial_field.dtype)
