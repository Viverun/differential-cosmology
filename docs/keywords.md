# Keywords and Concepts

This glossary explains common terms used in this project in plain language.

## Why this file exists

If you are new to cosmology, machine learning, or scientific computing, many words can feel overloaded or unclear. This page gives quick definitions and practical context for the exact terms used in this repository.

## Core Project Terms

### Gradient
A gradient tells you how much the loss changes if you slightly change each value in the initial field.

Why it matters here:
- We use gradients to update the initial field during reconstruction.
- Without gradients, inverse inference becomes slow trial-and-error.

### Field-level inference
Inferring an entire spatial field (all pixels/cells), not just a few summary numbers.

Why it matters here:
- You recover detailed structure, not only aggregate statistics.

### Initial field
A grid representing early-universe density fluctuations before structure formation.

### Forward model
A simulator that maps initial conditions to later-time structure.

In this project:
- Implemented in `src/diffcosmo/pm.py`.

### Inverse problem
Given observations, recover the unknown causes (here: initial field).

### Differentiable model
A model where outputs are smoothly connected to inputs so gradients can be computed through the full pipeline.

### Autodiff (automatic differentiation)
A method used by JAX to compute exact derivatives of code-defined computations.

### Backpropagation
Reverse-mode autodiff that propagates gradient information from loss back to inputs.

## Physics and Simulation Terms

### Density field
A grid showing over-dense and under-dense regions relative to average density.

### Power spectrum `P(k)`
A function describing how much fluctuation power exists at each spatial scale.

Interpretation:
- low `k`: large scales
- high `k`: small scales

### Fourier space
A representation of a field by spatial frequencies instead of positions.

Why used:
- Poisson solves and spectral operations are efficient in Fourier space.

### Particle-Mesh (PM)
A simulation approach where matter is represented with particles but forces are computed on a mesh/grid.

### PM-lite
A simplified PM approach used in this repository for a 2D differentiable MVP.

### CIC (Cloud-In-Cell)
A standard interpolation/deposition method to move information between particle positions and grid cells.

### Poisson equation
Relates density to gravitational potential. In this project, solved via FFT in the PM step.

### Periodic boundaries
Edges of the simulation wrap around so the box behaves like a repeating tile.

## Inference and Statistics Terms

### Observation model
Maps a simulated field to what a survey would measure, including noise and masking.

### Noise model
Assumptions about measurement noise (currently Gaussian in the toy setup).

### Mask
A spatial window that hides unobserved regions.

### Prior
Belief about plausible initial fields before seeing data.

### Likelihood
How probable the observed data is for a candidate initial field.

### Posterior
Updated belief after combining prior and likelihood.

Formula:
- `posterior ∝ likelihood × prior`

### MAP (Maximum A Posteriori)
The single best estimate under the posterior, found by minimizing a loss equivalent to negative log-posterior.

### Regularization
A term added to prevent unrealistic solutions and improve stability (here, spectral prior penalty).

## Optimization and Engineering Terms

### Loss function
A scalar objective that measures reconstruction quality and prior compliance.

### Learning rate
Step size for optimizer updates.

### Gradient clipping
Rescaling large gradients to prevent unstable updates.

### Backtracking line search
Reducing step size when a proposed optimizer step worsens the objective.

### Early stopping
Stopping optimization when progress stalls.

### JIT (Just-In-Time compilation)
Compiles functions for speed in JAX.

### Reproducibility
Ability to get consistent results from fixed seeds/config.

In practice:
- use config files
- set random seeds
- track metrics and outputs.

## Evaluation Terms

### Cross-correlation
A similarity score between true and reconstructed fields.

Typical interpretation:
- `1`: perfect agreement
- `0`: no linear relationship
- `-1`: perfect anti-correlation

### Power spectrum recovery
How closely the reconstructed field's `P(k)` matches the target/true field.

### Smoke test
A short end-to-end test that verifies the whole pipeline runs.

## Where to go next

- Beginner overview: `docs/beginner_guide.md`
- Hands-on tutorial: `docs/getting_started.md`
- Technical depth: `docs/overview.md`
