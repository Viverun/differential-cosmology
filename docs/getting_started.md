# Getting Started

This guide gets you from a fresh clone to a full 2D reconstruction run.

It is written for people who want to run the project first and learn details while doing.

## What you will do

1. Set up the environment with `uv`
2. Run tests
3. Run the toy 2D reconstruction pipeline
4. Inspect outputs and change one config knob

## 1. Setup

From the repository root:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync --active --extra dev
```

Quick sanity check:

```bash
uv run --active pytest -q
```

You should see all tests passing.

## 2. Run the default experiment

```bash
uv run --active python scripts/run_toy_2d.py --profile default
```

This executes the full pipeline:

- sample true initial field
- evolve forward with PM-lite
- apply mask/noise observation model
- reconstruct with MAP optimization
- save plots and arrays

By default, outputs go to `outputs/toy2d/`.

## 3. What to look at in the output

Primary figure:

- `outputs/toy2d/toy2d_summary.png`

It includes:

- true initial field
- initial guess
- reconstructed field
- residual map
- loss curve
- power spectrum comparison

Key printed metrics:

- `initial_loss` and `final_loss`
- `loss_reduction_fraction`
- `initial_corr` and `final_corr`
- `corr_improvement`

Interpretation:

- lower final loss is better
- higher final correlation is better
- positive correlation improvement means reconstruction learned useful structure

## 4. Run a faster debug profile

```bash
uv run --active python scripts/run_toy_2d.py --profile debug
```

Use this when iterating quickly.

## 5. Change one parameter and rerun

Edit `data/toy/config.yaml` and try one of these:

- increase `obs.noise_std` to make reconstruction harder
- reduce `opt.lr` if optimization looks unstable
- increase `sim.n_steps` to make the forward model more nonlinear

Then rerun and compare metrics/plots.

## 6. Minimal API walkthrough (optional)

The same pipeline can be called directly from Python.

```python
import jax.numpy as jnp
from diffcosmo.fields import generate_gaussian_field, power_law_power_spectrum
from diffcosmo.pm import evolve_pm
from diffcosmo.observe import apply_observation_model
from diffcosmo.loss import total_loss
from diffcosmo.inference import reconstruct_map

power_fn = lambda k: power_law_power_spectrum(k, amplitude=1.0, index=-2.0)
theta_true = generate_gaussian_field((64, 64), power_fn, seed=0)
delta_true = evolve_pm(theta_true, n_steps=10, dt=0.1, cosmology={"Omega_m": 0.3, "init_disp_scale": 0.25, "identity_mix": 0.85})
y_obs = apply_observation_model(delta_true, noise_std=0.1, seed=2)

theta_init = generate_gaussian_field((64, 64), power_fn, seed=1)
sim = {"n_steps": 10, "dt": 0.1, "checkpointing": "none", "cosmology": {"Omega_m": 0.3, "init_disp_scale": 0.25, "identity_mix": 0.85}}
prior = {"amplitude": 1.0, "index": -2.0, "lambda_prior": 1e-2, "eps": 1e-6}
obs = {"noise_std": 0.1, "mask": None}
opt = {"lr": 0.05, "max_iters": 100}

result = reconstruct_map(y_obs, theta_init, sim, prior, obs, opt)
print(result["loss_history"][-1])
```

## 7. Where to go next

- `docs/beginner_guide.md` for intuition first
- `docs/keywords.md` for terminology
- `docs/overview.md` for math and algorithms
- `docs/validation.md` for evaluation criteria

If something fails, include your command and full traceback when opening an issue.
