# Technical Overview

This document explains the core scientific and algorithmic ideas behind the current repository.

For a plain-language version, start with `docs/beginner_guide.md`.

## Core idea

Treat cosmological structure formation as a differentiable computation graph:

```text
theta0 (initial field) -> forward simulation -> synthetic observation -> loss
                                    ^                               |
                                    |----------- gradients ----------|
```

That lets us solve an inverse problem with gradient-based optimization.

## Forward and inverse problem in this repo

### Forward problem

Given an initial field `theta0`, predict a late-time field:

1. generate / load `theta0`
2. evolve with PM-lite dynamics
3. apply observation effects (mask, noise)

### Inverse problem

Given observed data `y_obs`, recover `theta0` that best explains it under a prior.

This is implemented as MAP reconstruction.

## Mathematical framing

### Posterior

`P(theta | y) proportional to P(y | theta) * P(theta)`

### MAP objective

We optimize a scalar loss of the form:

`L(theta) = data_term(theta) + lambda * prior_term(theta)`

In practice:

- `data_term` compares predicted vs observed field
- `prior_term` penalizes implausible spectral structure

### Why gradients help

The gradient `dL/dtheta` gives direction in a very high-dimensional space.
Without it, search is mostly brute force.

## Algorithmic components

## 1) Initial field generation

`fields.generate_gaussian_field` samples a Gaussian random field with a configurable power-law prior in Fourier space.

## 2) PM-lite forward model

`pm.evolve_pm` uses a differentiable 2D particle-mesh style update loop:

- lattice particles
- CIC deposit/interpolation
- FFT Poisson solve
- force update with periodic boundaries

## 3) Observation model

`observe.apply_observation_model` applies:

- optional spatial mask
- Gaussian noise

## 4) Loss construction

`loss.total_loss` combines:

- data misfit
- spectral prior penalty

## 5) MAP optimization

`inference.reconstruct_map` performs gradient-based updates with practical stability controls.

## Current implementation scope

What is production-ready in this repository today:

- validated 2D toy inverse pipeline
- deterministic config-driven experiments
- module-level and end-to-end tests

What is intentionally not done yet:

- full 3D production-scale implementation
- realistic end-to-end survey systematics
- full posterior sampling pipeline

## Why this design is practical

- It is small enough to debug.
- It is structured enough to extend.
- It keeps physics, inference, and diagnostics separated.

That combination is what makes the project usable for both learning and research iteration.

## Where to go next

- `docs/architecture.md` for code organization
- `docs/validation.md` for quality criteria
- `docs/getting_started.md` to run experiments quickly
