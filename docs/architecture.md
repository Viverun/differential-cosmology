# Software Architecture

This document explains how the code is organized today, why modules are split this way, and where to add new work.

If you are brand new, read `docs/beginner_guide.md` first.

## Architecture at a glance

```text
src/diffcosmo/
├── __init__.py
├── fields.py      # initial condition generation
├── pm.py          # differentiable 2D PM-lite forward model
├── observe.py     # mask and noise operators
├── loss.py        # MAP objective pieces
├── inference.py   # gradient-based reconstruction
└── utils.py       # spectral and analysis utilities
```

## Data flow

```text
theta0 (initial field)
  -> pm.evolve_pm(...)
  -> observe.apply_observation_model(...)
  -> loss.total_loss(...)
  -> inference.reconstruct_map(...)
  -> theta_map (reconstructed initial field)
```

The script `scripts/run_toy_2d.py` orchestrates this end to end.

## Design choices

### Functional style

Most code is written as pure-ish functions with explicit inputs and outputs.

Why:

- easier to test
- easier to reason about gradients
- fewer hidden side effects

### Clear module boundaries

Each module has one main responsibility.

- `fields.py`: build plausible starting fields
- `pm.py`: evolve physics forward
- `observe.py`: turn simulated fields into survey-like observations
- `loss.py`: define objective
- `inference.py`: optimize the objective
- `utils.py`: metrics and shared spectral helpers

### Reproducibility by config + seeds

Runs should be driven by `data/toy/config.yaml` and deterministic seeds whenever possible.

## Module details

## `fields.py`

Purpose:

- build Gaussian random initial fields from a target power spectrum

Key public function:

- `generate_gaussian_field(grid_shape, power_spectrum_fn, seed, dtype)`

Notes:

- generated field is centered and normalized
- currently focused on 2D use in this repository's MVP path

## `pm.py`

Purpose:

- implement differentiable 2D PM-lite evolution

Core components:

- lattice particle initialization
- CIC deposit/interpolation
- FFT Poisson solve for potential
- force computation and particle updates
- periodic boundaries

Key public function:

- `evolve_pm(initial_field, n_steps, dt, cosmology, checkpointing='none')`

## `observe.py`

Purpose:

- mimic imperfect observations

Key public functions:

- `make_rectangular_mask(grid_shape, frac)`
- `apply_observation_model(field, noise_std, mask=None, seed=0)`

Current scope:

- deterministic mask operator
- Gaussian additive noise

## `loss.py`

Purpose:

- build the MAP objective

Pieces:

- data misfit term
- spectral prior term
- weighted sum

Key public function:

- `total_loss(theta0, y_obs, sim_config, prior_config, obs_config)`

## `inference.py`

Purpose:

- optimize the initial field using gradients

Current implementation:

- Adam-style updates
- gradient clipping
- backtracking step control
- early stopping

Key public function:

- `reconstruct_map(y_obs, theta_init, sim_config, prior_config, obs_config, opt_config)`

## `utils.py`

Purpose:

- shared spectral and diagnostic helpers

Current functions include:

- `compute_k_values`
- `compute_power_spectrum`
- `cross_correlation`

## Runtime and experiment layer

### `scripts/run_toy_2d.py`

This is the canonical runnable pipeline for local experiments.

It loads config, runs simulation + reconstruction, saves plots/arrays, and prints key metrics.

### `data/toy/config.yaml`

Contains `default` and `debug` profiles.

Use this file for run configuration instead of hardcoding values in scripts.

## Testing structure

Tests live in `tests/` and cover:

- module-level behavior (fields, PM, observe, loss, inference, utils)
- end-to-end smoke execution

Core command:

```bash
uv run --active pytest -q
```

## Extension points

Likely next extension areas:

- 3D generalization of the PM pipeline
- richer observation operators
- posterior sampling beyond MAP

When extending, keep module boundaries intact and update tests with every change.
