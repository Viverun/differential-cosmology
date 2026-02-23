# Prerequisites

This project is approachable if you know basic Python and are willing to learn a bit of physics and optimization as you go.

You do not need to be an expert before starting.

## Quick readiness check

You are ready to begin if you can:

- run Python scripts from the terminal
- read basic array code (`numpy`/`jax.numpy` style)
- understand what a derivative/gradient is at a high level
- use Git for basic commit/push workflows

## What to learn first (high impact)

## 1) Python + arrays

Why:

- nearly everything in this repo is array-based computation

Focus on:

- indexing, shapes, broadcasting
- reading tracebacks and fixing simple bugs

## 2) Gradient-based optimization

Why:

- reconstruction is done by minimizing a loss with gradients

Focus on:

- loss function
- learning rate
- gradient descent intuition

## 3) Fourier intuition

Why:

- PM and spectral priors use Fourier-space operations

Focus on:

- what frequency space means
- rough meaning of power spectrum `P(k)`

## 4) Basic probabilistic inference terms

Why:

- MAP objective combines data fit and prior belief

Focus on:

- likelihood
- prior
- posterior

## Helpful but optional

- deeper cosmology background
- advanced Bayesian methods
- GPU performance tuning
- advanced numerical integration theory

These help as the project scales to 3D and richer models, but they are not required for initial contribution.

## Practical tooling knowledge

Recommended comfort level with:

- `uv` for environment and run commands
- `pytest` for tests
- `git` for branch + commit workflow

Core commands:

```bash
uv sync --active --extra dev
uv run --active pytest -q
uv run --active python scripts/run_toy_2d.py --profile default
```

## Learning strategy that works well

Do this in loops:

1. run the pipeline
2. change one config parameter
3. observe what changed
4. read only the theory needed to explain that change

That keeps learning grounded and avoids getting stuck in abstract study.

## Suggested next doc after this

- `docs/getting_started.md`
- `docs/keywords.md`
