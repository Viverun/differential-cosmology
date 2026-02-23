# Differentiable Cosmology: The Inverse Universe

A research codebase for **field-level inverse cosmology**:

- Generate primordial density fields
- Evolve them with a differentiable 2D PM-lite model
- Build noisy/masked observations
- Reconstruct initial conditions with gradient-based MAP inference

The long-term goal is 3D differentiable reconstruction and posterior inference. The current codebase is a validated 2D MVP.

## Start Here First

If you are new to this topic, start with:

- `docs/index.md` - visual documentation map with reading paths by skill level

Then read these two files:

- `docs/beginner_guide.md` - full plain-language project walkthrough
- `docs/keywords.md` - glossary of terms like `gradient`, `field-level`, `MAP`, and `power spectrum`

## What Has Been Implemented So Far

### Core 2D pipeline
- `fields.py`: Gaussian initial condition sampling with configurable power-law power spectrum.
- `pm.py`: Differentiable 2D PM-lite evolution with
  - lattice particles
  - CIC deposit/interpolation
  - FFT Poisson solve
  - force-based updates with periodic boundaries.
- `observe.py`: Observation model with rectangular survey masks and Gaussian noise.
- `loss.py`: MAP objective = data misfit + spectral Gaussian prior.
- `inference.py`: Adam-based MAP reconstruction with gradient clipping, backtracking, and early stopping.
- `utils.py`: `compute_k_values`, radial power spectrum estimation, and cross-correlation diagnostics.

### Experiment and configuration
- `scripts/run_toy_2d.py` runs the full end-to-end experiment and writes diagnostics/figures.
- `data/toy/config.yaml` contains `default` and `debug` profiles.

### Validation and tests
- Unit tests now cover fields, PM, observation, loss, inference, and utility functions.
- End-to-end smoke test added.
- Current test status: `20 passed`.

### Notebooks
- `notebooks/00_toy_2d_forward.ipynb`
- `notebooks/01_toy_2d_inverse.ipynb`
- `notebooks/utils.ipynb`

### Visual example output
- `docs/assets/toy2d_summary_example.png` (generated from `scripts/run_toy_2d.py`)

## Why This Project Matters

Traditional cosmology workflows rely heavily on repeated forward simulations. This project explores a differentiable alternative:

1. Build a differentiable forward model.
2. Backpropagate through physics.
3. Infer initial conditions directly with gradients.

This opens a practical path to faster and richer field-level inference.

## Quick Start (`uv`)

```bash
git clone <your-repo-url>
cd differentiable-cosmology

# Create or activate your project environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Sync dependencies from pyproject (dev tools included)
uv sync --active --extra dev

# Run tests
uv run --active pytest -q

# Run toy experiment
uv run --active python scripts/run_toy_2d.py --profile default
```

Outputs are written under `outputs/toy2d/` by default.

## Documentation

Start here based on your background:

- `docs/index.md` - visual docs map and recommended reading order
- `docs/beginner_guide.md` - complete plain-language guide (physics, math, data, engineering, and roadmap)
- `docs/keywords.md` - keyword glossary used throughout this project
- `docs/faq.md` - quick newcomer Q&A
- `docs/getting_started.md` - hands-on usage tutorial
- `docs/overview.md` - technical overview
- `docs/architecture.md` - software architecture and module structure
- `docs/validation.md` - metrics and validation strategy
- `docs/prerequisites.md` - background and learning resources

## Repository Structure

```text
src/diffcosmo/      Core library
scripts/            Runnable scripts
tests/              Unit + smoke tests
notebooks/          Interactive examples
data/               Configurations
docs/               Documentation
```

## Current Scope and Next Steps

### In scope now
- 2D differentiable toy reconstruction
- MAP inference and diagnostics
- Reproducible local experimentation

### Next
- Robust 3D extension on GPU
- More realistic observation operators
- Posterior sampling (MCMC / amortized methods)

## Contributing

See `CONTRIBUTING.md`.

## License

MIT (`LICENSE`).
