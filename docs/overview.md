# Technical Overview

This document provides technical depth on the architecture, algorithms, and design decisions behind the Differentiable Cosmology project. It's intended for contributors, advanced users, and researchers who want to understand or extend the system.

---

## Core Concept

The project treats cosmological structure formation as a **differentiable computation graph**, enabling gradient-based inference of initial conditions from observations. Instead of forward-only simulation, we maintain differentiability throughout the pipeline:

```
Initial Field (θ) → Forward Model (f) → Observations (y)
                    ↓ (backprop)
                Gradients (∂L/∂θ)
```

This allows us to solve the inverse problem: given observations `y`, find the initial conditions `θ` that maximize `P(θ|y)` using gradient-based optimization or sampling.

---

## System Architecture

### Component Hierarchy

The system is organized into three main layers:

**1. Forward Model Layer** (`src/diffcosmo/fields.py`, `pm.py`, `observe.py`)
- Generates initial density fields from cosmological power spectra
- Evolves fields forward in time using particle-mesh dynamics
- Applies observational operators (masks, noise, selection effects)

**2. Inference Layer** (`src/diffcosmo/inference.py`, `loss.py`)
- Defines likelihoods and priors
- Implements gradient-based optimizers (MAP estimation)
- Provides sampling algorithms (MCMC, learned samplers)

**3. Utilities Layer** (`src/diffcosmo/utils.py`)
- FFT operations and spectral methods
- Checkpointing and memory management
- Visualization and diagnostics

### Data Flow

```
1. Initial Conditions
   └─> fields.py: generate_gaussian_field(power_spectrum, seed)
       └─> returns: δ₀(x) [density field at z_initial]

2. Forward Evolution
   └─> pm.py: evolve_pm(δ₀, cosmology, n_steps)
       └─> returns: δ_final(x) [density field at z_obs]

3. Observation Model
   └─> observe.py: apply_survey(δ_final, mask, noise, bias)
       └─> returns: y_obs [mock observations]

4. Loss Computation
   └─> loss.py: compute_loss(y_obs, y_target, prior)
       └─> returns: L [scalar loss + gradients]

5. Inference
   └─> inference.py: optimize(θ_init, loss_fn)
       └─> returns: θ_MAP [reconstructed initial field]
```

---

## Mathematical Foundation

### The Forward Problem

The forward model consists of three stages:

**Stage 1: Initial Conditions**

Generate a Gaussian random field with specified power spectrum:
```
δ₀(k) ~ N(0, P(k))
```
where `P(k)` is the linear matter power spectrum at initial redshift.

**Stage 2: Gravitational Evolution**

Evolve the density field using the particle-mesh method, which approximates the collisionless Boltzmann equation:
```
∂ρ/∂t + ∇·(ρv) = 0           (continuity)
∂v/∂t + (v·∇)v = -∇Φ          (Euler)
∇²Φ = 4πGρ                    (Poisson)
```

In comoving coordinates and with appropriate expansions for cosmology, this reduces to a set of ODEs that can be integrated using leap-frog or higher-order symplectic methods.

**Stage 3: Observation Operator**

Map the evolved density field to observable quantities:
```
y = M(δ_final) + noise
```
where `M` includes:
- Halo/galaxy sampling (e.g., HOD model)
- Survey geometry and masking
- Redshift-space distortions
- Photometric/spectroscopic selection

### The Inverse Problem

Given observations `y`, infer the posterior over initial conditions:
```
P(θ|y) ∝ P(y|θ) P(θ)
```

**Likelihood:**
```
P(y|θ) = N(y | f(θ), Σ_noise)
```
where `f(θ)` is the forward model.

**Prior:**
```
P(θ) = N(θ | 0, P(k))
```
reflecting our knowledge of the initial power spectrum.

**MAP Estimation:**

Find the maximum a posteriori estimate:
```
θ_MAP = argmax_θ [log P(y|θ) + log P(θ)]
```

This optimization problem is solved using gradient descent with:
```
∇_θ L = ∇_θ log P(y|θ) + ∇_θ log P(θ)
```

Gradients are computed via automatic differentiation through the entire forward model.

---

## Key Algorithms

### 1. Particle-Mesh Evolution

The PM method discretizes the continuous density field on a grid and uses FFTs for efficient force computation:

**Algorithm:**
```
1. Initialize particles on grid with positions x_i and velocities v_i
2. For each timestep:
   a. Assign particle densities to grid: ρ(x_grid) = Σ_i W(x_i - x_grid)
   b. Compute potential via FFT: Φ(k) = -4πG ρ(k) / |k|²
   c. Compute forces: F(x_grid) = -∇Φ(x_grid)
   d. Interpolate forces to particles: F_i = Σ_grid W(x_i - x_grid) F(x_grid)
   e. Update velocities: v_i = v_i + F_i Δt
   f. Update positions: x_i = x_i + v_i Δt
```

**Differentiability:** All operations (assignment, FFT, interpolation) have well-defined gradients, making the entire evolution differentiable.

### 2. Gradient Computation

Gradients are computed using reverse-mode automatic differentiation (backpropagation):

**Forward pass:** Compute loss by running the full simulation
**Backward pass:** Accumulate gradients ∂L/∂θ by traversing the computation graph in reverse

**Memory challenge:** Backpropagation through `n` timesteps requires storing `O(n)` intermediate states.

**Solution:** Gradient checkpointing
- Store states at coarse intervals
- Recompute intermediate states during backward pass
- Trade compute for memory: `O(√n)` memory, `O(n)` extra compute

### 3. MAP Optimization

We use gradient-based optimizers with preconditioning:

**Basic gradient descent:**
```
θ_{t+1} = θ_t - α ∇_θ L(θ_t)
```

**With momentum (Adam):**
```
m_t = β₁ m_{t-1} + (1-β₁) ∇_θ L
v_t = β₂ v_{t-1} + (1-β₂) (∇_θ L)²
θ_{t+1} = θ_t - α m_t / (√v_t + ε)
```

**Preconditioning:** Scale gradients by approximate inverse Hessian to account for different scales in Fourier space.

### 4. Posterior Sampling (Future)

Beyond MAP estimation, we plan two approaches for posterior sampling:

**Gradient-informed MCMC:**
- Use Hamiltonian Monte Carlo (HMC) to sample high-dimensional posteriors
- Gradients guide proposals, improving mixing

**Amortized inference:**
- Train normalizing flows or score-based models: `q_φ(θ|y)`
- One forward pass produces approximate posterior samples
- Amortizes cost across many observations

---

## Design Decisions & Rationale

### Why JAX?

**Automatic differentiation:** Native support for reverse-mode AD through arbitrary Python code

**Performance:** JIT compilation, XLA optimization, automatic vectorization and parallelization

**Functional paradigm:** Pure functions make checkpointing and gradient computation cleaner

**Hardware flexibility:** Runs on CPU, GPU, TPU with minimal code changes

**Ecosystem:** Growing astro/cosmo libraries (JAX-Cosmo, JaxPM)

**Alternative considered:** PyTorch (more mature ecosystem, but less functional, less efficient for scientific computing)

### Why Particle-Mesh?

**Differentiability:** All operations (grid assignment, FFTs, interpolation) have well-defined gradients

**Efficiency:** `O(N log N)` via FFTs vs. `O(N²)` for direct N-body

**Memory:** Linear scaling in resolution enables larger volumes

**Good enough:** Captures large-scale structure accurately; small-scale accuracy less critical for initial condition reconstruction

**Limitations:** Cannot resolve small-scale phase-space structure, lacks baryonic physics

**Future:** Can extend to differentiable hydrodynamics for baryonic realism

### Checkpointing Strategy

**Problem:** Backpropagating through `T` timesteps requires `O(T)` memory to store intermediate states.

**Solution hierarchy:**

1. **None:** Store all states (fast but memory-intensive)
2. **Uniform checkpointing:** Store every `k`-th state, recompute in between
3. **Logarithmic checkpointing:** Store at exponentially spaced intervals
4. **Reversible integration:** Use symplectic integrators that can be inverted exactly

**Current approach:** User-configurable with sensible defaults (uniform checkpointing for toy problems, logarithmic for production)

### Observation Model Philosophy

**Progressive realism:** Start simple, add complexity incrementally

**Phase 1:** Point process (Poisson sampling) with Gaussian noise
**Phase 2:** Survey masks, redshift-space distortions (RSD)
**Phase 3:** Halo occupation distribution (HOD), realistic selection functions
**Phase 4:** Baryonic effects, photo-z errors, systematic uncertainties

**Rationale:** Validate at each level before adding complexity. Model mismatch is a bigger risk than initial simplicity.

---

## Technical Constraints & Tradeoffs

### Memory vs. Resolution

**Grid size:** `N³` cells requires `O(N³)` memory for density field

**Particles:** Typically `N_p = N_cells` for adequate sampling

**Gradient storage:** Without checkpointing, needs `O(T × N³)` for `T` timesteps

**Practical limits:**
- 2D toy: 128² to 512² cells (MB-scale, runs on CPU)
- 3D small: 64³ to 128³ cells (GB-scale, single GPU)
- 3D medium: 256³ to 512³ cells (tens of GB, multi-GPU)
- Production: 1024³+ cells (distributed, HPC required)

### Speed vs. Accuracy

**Timestep size:** Larger steps = faster but less accurate evolution

**Grid resolution:** Coarser grid = faster FFTs but worse force resolution

**Integration scheme:** Higher-order symplectic = fewer steps but more ops per step

**Typical choices:**
- Toy problems: Leap-frog integration, `dt = 0.1 a/H(a)`
- Production: 2nd or 4th order symplectic, adaptive timesteps

### Inference Cost

**Forward pass:** `O(T × N³ log N)` — dominated by FFTs

**Backward pass:** Same cost + gradient accumulation overhead (~2× total)

**MAP optimization:** 10-100 gradient steps typical

**MCMC sampling:** 1000-10000 steps for convergence

**Amortized sampling:** Training cost (high) amortized over inference (low)

**Implication:** MAP is practical for high-res; full MCMC limited to smaller problems

---

## Extension Points

The codebase is designed for extensibility in several key areas:

### 1. Custom Forward Models

Replace `pm.py` with alternative evolution schemes:

```python
def evolve_custom(initial_field, cosmology, n_steps):
    """Custom evolution function.
    
    Must be differentiable and return final density field.
    """
    # Your implementation here
    return final_field
```

**Examples:**
- Lagrangian perturbation theory (LPT)
- Differentiable hydrodynamics
- Neural emulators trained on high-fidelity sims

### 2. Custom Observation Models

Extend `observe.py` with domain-specific operators:

```python
def apply_custom_survey(field, survey_params):
    """Apply custom survey characteristics.
    
    Args:
        field: evolved density field
        survey_params: survey-specific parameters
        
    Returns:
        observed quantities with realistic systematics
    """
    # Your implementation here
    return observations
```

**Examples:**
- Weak lensing observables (shear, convergence)
- 21cm intensity mapping
- CMB lensing
- Spectroscopic redshift surveys with realistic selection

### 3. Custom Priors

Modify `loss.py` to incorporate domain knowledge:

```python
def custom_prior(theta, prior_params):
    """Custom prior distribution.
    
    Can encode physical constraints or learned priors.
    """
    return log_prior
```

**Examples:**
- Non-Gaussian priors from simulations
- Physically motivated constraints (positivity, smoothness)
- Hierarchical priors on cosmological parameters

### 4. Custom Samplers

Add new inference methods in `inference.py`:

```python
def custom_sampler(loss_fn, initial_state, sampler_params):
    """Custom posterior sampling algorithm.
    
    Should return samples from P(θ|y).
    """
    # Your implementation here
    return samples
```

**Examples:**
- Variational inference
- Neural posterior estimation
- Ensemble methods
- Sequential Monte Carlo

---

## Performance Considerations

### Profiling Hotspots

Typical computational breakdown:

- **FFTs:** 40-60% of forward pass time
- **Particle assignment/interpolation:** 20-30%
- **Gradient accumulation:** 10-20%
- **Everything else:** <10%

**Optimization priorities:**
1. Efficient FFT library (use FFTW backend in JAX)
2. Vectorized assignment/interpolation kernels
3. Minimize data movement between CPU/GPU
4. Batch multiple forward passes when possible

### Scaling Strategy

**Single GPU:**
- 128³ cells: ~few seconds per forward pass (A100)
- 256³ cells: ~10-30 seconds per forward pass
- 512³ cells: ~1-3 minutes per forward pass

**Multi-GPU:**
- Spatial domain decomposition via JAX `pmap`
- FFTs across GPU boundaries (more complex)
- Linear scaling up to ~8-16 GPUs

**HPC/Cluster:**
- Use JAX distributed primitives
- MPI-style communication for large-scale FFTs
- Checkpoint to persistent storage
- Job orchestration for parameter sweeps

### Memory Optimization

**Techniques:**
1. **Mixed precision:** Use float32 for storage, float64 for critical ops
2. **Gradient checkpointing:** Recompute instead of storing
3. **Lazy evaluation:** Compute fields on-demand
4. **Compressed storage:** Store Fourier modes, reconstruct real-space as needed

---

## Validation Strategy

See [`validation.md`](validation.md) for full details on metrics and tests. Key principles:

**Unit tests:** Each component tested in isolation
**Integration tests:** End-to-end forward+inverse on known cases
**Convergence tests:** Results stable to resolution, timestep, random seed
**Physical tests:** Power spectra, correlation functions match theory
**Statistical tests:** Posterior coverage, calibration checks

---

## Known Limitations

### Current Implementation

- **2D only:** 3D PM evolution under development
- **Simple observation model:** No realistic survey characteristics yet
- **MAP only:** Posterior sampling infrastructure planned but not implemented
- **Single GPU:** Multi-GPU support requires architecture changes

### Fundamental Constraints

- **PM limitations:** Cannot capture shell-crossing, small-scale phase space
- **Gaussian initial conditions:** Non-Gaussianity from inflation not yet supported
- **Dark matter only:** No baryonic physics or feedback
- **Computational cost:** Full posterior inference expensive at high resolution

### Future Work

These limitations define our roadmap. See README and CHANGELOG for current progress.

---

## References & Related Work

**Differentiable simulation:**
- FlowPM (Modi et al. 2021) - JAX-based differentiable PM
- JaxPM (Lanzieri et al. 2024) - Particle mesh in JAX
- Differentiable cosmology review (Villaescusa-Navarro et al. 2022)

**Field-level inference:**
- BORG (Jasche & Wandelt 2013) - Bayesian large-scale structure inference
- ELUCID (Wang et al. 2016) - Constrained reconstruction
- Differentiable inference (Li et al. 2021)

**Particle-mesh methods:**
- Hockney & Eastwood (1988) - Classic PM text
- PMFAST (Merz et al. 2005)
- Gadget-2 (Springel 2005) - P³M hybrid

---

## Contributing to This Document

This overview evolves with the codebase. When adding major features:

1. Update the relevant architecture section
2. Document key design decisions and tradeoffs  
3. Add extension points if introducing new abstractions
4. Update limitations and known issues

Keep it technical but accessible. The goal is to help contributors understand *why* things work the way they do, not just *what* they do.