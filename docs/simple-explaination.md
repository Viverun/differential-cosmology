# Simple explanation: Differentiable Cosmology in plain language

## One-line idea

We want to look at the universe we observe today and reconstruct the likely initial matter pattern that created it.

## The core intuition

Think of cosmic structure like ripples in water.
At the beginning, the ripples are tiny.
Over time, gravity amplifies them into filaments, clusters, and voids.

Traditional workflow is mostly:

- guess an initial pattern
- run a simulation forward
- compare with observations
- repeat, repeat, repeat

This project keeps the simulation differentiable, so we can compute gradients and move the guess in a smarter direction instead of searching blindly.

## Why this matters

- You recover full spatial structure, not only a few summary numbers.
- You can use optimization methods that are much more directed than brute-force trial and error.
- You get a practical bridge between physics simulation and modern ML tooling.

## What this system is made of

1. **Initial field generator**
   It creates a plausible early-universe density map using a power-spectrum prior.

2. **Forward physics model (PM-lite)**
   It evolves that map forward in time with a differentiable particle-mesh style solver.

3. **Observation model**
   It applies effects like masking and noise to imitate imperfect measurements.

4. **Inference loop**
   It compares prediction vs observation, computes loss and gradients, and updates the initial field.

5. **Diagnostics and validation**
   It checks whether reconstruction improved using loss curves, cross-correlation, and power spectrum comparisons.

## What is already implemented here

This repository already has a working 2D MVP:

- differentiable 2D PM-lite forward model
- mask + Gaussian noise observation model
- MAP reconstruction with gradient-based optimization
- tests and an end-to-end toy pipeline (`scripts/run_toy_2d.py`)

So this is not just a concept note; you can run it and inspect real outputs.

## What comes next

- extend robustly from 2D to 3D
- improve realism of observation effects
- move from one best-fit map (MAP) toward posterior sampling
- scale experiments to stronger GPU setups

## The hard parts (realistically)

- **Numerical stability:** gradients can explode or become unstable if the pipeline is not carefully constrained.
- **Memory pressure:** backpropagating through time-evolution is expensive.
- **Model mismatch:** real observations are messy; simplistic assumptions can bias reconstruction.

## Three practical moves you can make today

1. Run the full toy pipeline and inspect the outputs.
   `uv run --active python scripts/run_toy_2d.py --profile default`

2. Open the notebooks and change one important knob (noise level, steps, or learning rate), then rerun.

3. Write a short experiment note: what improved, what failed, and one hypothesis for why.

That habit builds scientific progress much faster than random feature adding.
