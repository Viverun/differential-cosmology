# Software Architecture

This document describes the software architecture of the Differentiable Cosmology codebase. It's intended for contributors who want to understand how the code is organized, how modules interact, and where to make changes.

For the scientific/algorithmic architecture, see [`overview.md`](overview.md).

---

## Design Philosophy

The codebase follows these core principles:

**1. Functional over Object-Oriented**
- Pure functions with explicit inputs/outputs
- Minimal mutable state
- Enables JAX transformations (jit, grad, vmap)
- Makes testing and debugging easier

**2. Modular and Composable**
- Each module has a single clear responsibility
- Functions can be composed in different ways
- Easy to swap implementations (e.g., different PM solvers)

**3. Separation of Concerns**
- Forward model (physics) separate from inference (optimization/sampling)
- I/O separate from computation
- Configuration separate from implementation

**4. Autodiff-Friendly**
- All operations differentiable by default
- Avoid in-place modifications
- Use JAX primitives, not NumPy
- Clear boundaries for non-differentiable ops (I/O, plotting)

**5. Testable by Design**
- Small, focused functions
- Deterministic (controlled randomness via seeds)
- Easy to create test cases
- Clear success criteria

---

## Module Organization

### High-Level Structure

```
src/diffcosmo/
├── __init__.py          # Package initialization, public API
├── fields.py           # Initial condition generation
├── pm.py               # Particle-mesh evolution
├── observe.py          # Observation operators
├── loss.py             # Loss functions and priors
├── inference.py        # MAP optimization and sampling
└── utils.py            # Shared utilities (FFT, I/O, plotting)
```

### Module Dependency Graph

```
fields.py  ──┐
             ├──> pm.py ──> observe.py ──> loss.py ──> inference.py
utils.py  ───┴────────────────────────────────────────────┘
                          │
                          └──> (scripts, notebooks)
```

**Key points:**
- `utils.py` is a leaf dependency (imported by all)
- `fields.py` has no internal dependencies
- Data flows left-to-right: fields → PM → observe → loss → inference
- Scripts/notebooks orchestrate but don't contain logic

---

## Module Details

### `fields.py` — Initial Condition Generation

**Purpose:** Generate Gaussian random fields with specified power spectra.

**Key Functions:**

```python
def generate_power_spectrum(k, cosmology_params):
    """Compute P(k) from cosmological parameters.
    
    Args:
        k: wavenumber array (1D)
        cosmology_params: dict with Omega_m, sigma_8, etc.
    
    Returns:
        P_k: power spectrum array (same shape as k)
    """
    
def generate_gaussian_field(grid_shape, power_spectrum, seed):
    """Generate Gaussian random field with given power spectrum.
    
    Args:
        grid_shape: tuple (N, N) for 2D or (N, N, N) for 3D
        power_spectrum: function k -> P(k)
        seed: random seed for reproducibility
    
    Returns:
        field: complex array in Fourier space with shape grid_shape
    """
    
def field_to_real(field_fourier):
    """Convert Fourier-space field to real space.
    
    Args:
        field_fourier: complex array
    
    Returns:
        field_real: real-valued density field
    """
```

**Data Structures:**
- Density fields are 2D/3D JAX arrays
- Fourier-space fields are complex128
- Real-space fields are float64 (or float32 for memory)
- Power spectra are 1D arrays or callable functions

**Design Notes:**
- All randomness controlled by explicit seeds
- Works in Fourier space for efficiency
- Power spectrum is a separate function (can be swapped)
- No cosmology-specific code (just math)

---

### `pm.py` — Particle-Mesh Evolution

**Purpose:** Evolve density fields forward in time using particle-mesh dynamics.

**Key Functions:**

```python
def initialize_particles(density_field):
    """Initialize particle positions and velocities from density field.
    
    Args:
        density_field: initial density on grid
    
    Returns:
        particles: dict with 'positions' and 'velocities'
    """

def pm_step(particles, grid_shape, dt, cosmology):
    """Single PM timestep: assign → force → kick → drift.
    
    Args:
        particles: dict with positions/velocities
        grid_shape: grid resolution
        dt: timestep size
        cosmology: cosmological parameters
    
    Returns:
        particles: updated state
    """

def evolve_pm(initial_field, n_steps, cosmology, checkpointing='uniform'):
    """Full PM evolution with gradient checkpointing.
    
    Args:
        initial_field: starting density field
        n_steps: number of timesteps
        cosmology: parameters
        checkpointing: 'none', 'uniform', or 'logarithmic'
    
    Returns:
        final_field: evolved density field
        trajectory: (optional) saved states for visualization
    """
```

**Internal Structure:**

```
pm.py
├── Grid Operations
│   ├── assign_particles_to_grid()    # CIC assignment
│   └── interpolate_force_to_particles()  # CIC interpolation
├── Force Computation
│   ├── compute_potential_fft()       # Solve Poisson via FFT
│   └── compute_force_from_potential()  # Gradient of potential
├── Integration
│   ├── kick_step()                   # Update velocities
│   └── drift_step()                  # Update positions
└── Checkpointing
    ├── checkpoint_states()           # Save at intervals
    └── recompute_from_checkpoint()   # Recompute for backprop
```

**Data Structures:**
- Particles: `dict` with keys `'positions'`, `'velocities'` (both N×D arrays)
- Grid: 2D/3D array for density, potential, force fields
- Trajectory: list of particle states (for visualization/debugging)

**Design Notes:**
- Fully differentiable via JAX autodiff
- Checkpointing configurable (memory vs. compute trade-off)
- Symplectic integration (energy-conserving)
- Grid operations use JAX primitives for GPU acceleration
- Periodic boundary conditions built-in

---

### `observe.py` — Observation Operators

**Purpose:** Convert evolved density fields into realistic mock observations.

**Key Functions:**

```python
def sample_galaxies_poisson(density_field, n_galaxies, seed):
    """Sample galaxy positions from density field via Poisson sampling.
    
    Args:
        density_field: matter density (normalized)
        n_galaxies: expected number of galaxies
        seed: random seed
    
    Returns:
        galaxy_positions: N×D array of positions
    """

def apply_survey_mask(galaxy_positions, mask_function):
    """Apply geometric survey mask.
    
    Args:
        galaxy_positions: positions
        mask_function: callable returning True if inside survey
    
    Returns:
        masked_positions: positions inside survey volume
    """

def add_observation_noise(galaxy_positions, noise_level, seed):
    """Add position measurement noise.
    
    Args:
        galaxy_positions: true positions
        noise_level: standard deviation
        seed: random seed
    
    Returns:
        noisy_positions: observed positions
    """

def compute_galaxy_density_field(galaxy_positions, grid_shape):
    """Convert galaxy positions back to grid for comparison.
    
    Args:
        galaxy_positions: observed positions
        grid_shape: target grid resolution
    
    Returns:
        observed_field: galaxy density on grid
    """
```

**Design Notes:**
- Each operator is independently differentiable
- Operators can be composed: density → galaxies → mask → noise
- Observation model complexity is modular (add features incrementally)
- All randomness via explicit seeds

---

### `loss.py` — Loss Functions and Priors

**Purpose:** Define likelihood and prior terms for inference.

**Key Functions:**

```python
def gaussian_likelihood(predicted, observed, noise_covariance):
    """Negative log-likelihood for Gaussian noise.
    
    Args:
        predicted: forward model output
        observed: target observations
        noise_covariance: measurement noise (scalar or matrix)
    
    Returns:
        nll: negative log P(observed | predicted)
    """

def power_spectrum_prior(field_fourier, target_power_spectrum):
    """Prior enforcing power spectrum.
    
    Args:
        field_fourier: density field in Fourier space
        target_power_spectrum: expected P(k)
    
    Returns:
        log_prior: log P(field | P(k))
    """

def total_loss(theta, forward_model, observations, prior_params):
    """Combined loss: negative log posterior.
    
    Args:
        theta: initial conditions (parameters to infer)
        forward_model: callable that runs simulation
        observations: target data
        prior_params: prior configuration
    
    Returns:
        loss: -log P(theta | observations)
        aux: auxiliary info (diagnostics)
    """
```

**Design Pattern:**

```python
# Loss functions return (loss, aux) for gradient + logging
def loss_fn(params):
    prediction = forward_model(params)
    likelihood_term = gaussian_likelihood(prediction, data)
    prior_term = power_spectrum_prior(params)
    
    loss = likelihood_term + prior_term
    aux = {
        'likelihood': likelihood_term,
        'prior': prior_term,
        'prediction': prediction  # for visualization
    }
    return loss, aux
```

**Design Notes:**
- Loss functions are pure (no side effects)
- Return auxiliary data for logging (via `value_and_grad`)
- Priors are separate functions (easy to swap)
- Support for different noise models (diagonal, full covariance)

---

### `inference.py` — Optimization and Sampling

**Purpose:** Implement MAP optimization and posterior sampling algorithms.

**Key Functions:**

```python
def optimize_map(loss_fn, initial_params, optimizer='adam', 
                 n_steps=1000, learning_rate=1e-3):
    """MAP optimization via gradient descent.
    
    Args:
        loss_fn: function returning (loss, aux)
        initial_params: starting point
        optimizer: 'adam', 'sgd', or 'lbfgs'
        n_steps: optimization steps
        learning_rate: step size
    
    Returns:
        params_map: optimized parameters
        history: loss trajectory and diagnostics
    """

def sample_hmc(loss_fn, initial_params, n_samples=1000, 
               step_size=0.01, n_leapfrog=10):
    """Hamiltonian Monte Carlo sampling.
    
    Args:
        loss_fn: negative log posterior
        initial_params: starting point
        n_samples: number of samples
        step_size: HMC step size
        n_leapfrog: leapfrog steps per sample
    
    Returns:
        samples: array of posterior samples
        diagnostics: acceptance rate, ESS, etc.
    """

def train_flow_sampler(flow_model, training_data, n_epochs=100):
    """Train normalizing flow for amortized inference.
    
    Args:
        flow_model: invertible neural network
        training_data: (observations, true_params) pairs
        n_epochs: training iterations
    
    Returns:
        trained_flow: flow that maps noise → posterior samples
        training_history: loss curves
    """
```

**Internal Structure:**

```
inference.py
├── Optimizers
│   ├── adam_optimizer()
│   ├── sgd_optimizer()
│   └── lbfgs_optimizer()
├── Gradient Preprocessing
│   ├── gradient_clipping()
│   └── precondition_gradient()  # Scale by inverse Hessian approx
├── MCMC Samplers
│   ├── hmc_kernel()
│   ├── nuts_kernel()
│   └── adaptive_step_size()
└── Learned Samplers
    ├── flow_architecture()      # Normalizing flow definition
    └── score_model()            # Score-based diffusion model
```

**Design Notes:**
- Uses optax for optimizers (standard JAX library)
- Supports multiple optimization algorithms
- HMC/NUTS for small-scale posterior sampling
- Flow/score models for amortized inference (Phase 3)
- All samplers return diagnostics for debugging

---

### `utils.py` — Shared Utilities

**Purpose:** Common functions used across modules.

**Key Functions:**

```python
# FFT Operations
def fft(x, axes=None):
    """FFT wrapper with consistent conventions."""
    
def ifft(x, axes=None):
    """Inverse FFT wrapper."""
    
def compute_power_spectrum(field):
    """Compute power spectrum from field."""

# I/O
def save_checkpoint(state, filepath):
    """Save state dict to disk."""
    
def load_checkpoint(filepath):
    """Load state dict from disk."""

# Visualization
def plot_field_2d(field, title=''):
    """Plot 2D density field."""
    
def plot_power_spectrum(k, P_k, P_k_target=None):
    """Plot power spectrum comparison."""
    
def plot_loss_history(history):
    """Plot optimization trajectory."""

# Math Utilities
def compute_cross_correlation(field1, field2):
    """Cross-correlation coefficient r(k)."""
    
def bin_power_spectrum(k_values, P_values, n_bins):
    """Radial binning of power spectrum."""
```

**Design Notes:**
- Thin wrappers around JAX/matplotlib for consistency
- I/O uses JAX-native formats (npz, pickle of pytrees)
- Plotting functions are NOT differentiable (use outside grad context)
- Math utilities are differentiable

---

## Data Flow Architecture

### Forward Pass (Simulation)

```
1. Configuration
   └─> config.yaml: {cosmology, grid_size, n_steps, ...}

2. Initial Conditions
   └─> fields.generate_gaussian_field(seed)
       └─> field_fourier [complex array]

3. Evolution
   └─> pm.evolve_pm(field_fourier)
       └─> particles at each timestep
       └─> final_field [real array]

4. Observation
   └─> observe.sample_galaxies_poisson(final_field)
       └─> observe.apply_survey_mask(galaxies)
       └─> observe.add_observation_noise(galaxies)
       └─> observed_field [real array]

5. Comparison
   └─> loss.gaussian_likelihood(observed_field, target_field)
       └─> scalar loss
```

### Backward Pass (Inference)

```
1. Loss Computation
   └─> loss = loss_fn(theta)  # theta = initial field params

2. Gradient Computation
   └─> grad = jax.grad(loss_fn)(theta)
       └─> Backprop through entire forward model
       └─> Checkpoints used to recompute states

3. Optimization Step
   └─> theta_new = theta - learning_rate * grad
       └─> Update parameters

4. Iterate
   └─> Repeat until convergence
```

### Key Observation

**The entire forward model is one differentiable function:**

```python
def forward_model(theta):
    field = theta_to_field(theta)          # fields.py
    evolved = evolve_pm(field, ...)        # pm.py
    observed = apply_observations(evolved) # observe.py
    return observed

loss = lambda theta: gaussian_likelihood(forward_model(theta), data)
grad_fn = jax.grad(loss)
```

This is the core architectural insight that enables gradient-based inference.

---

## Configuration System

Configuration is handled via YAML files + Python dicts.

**Structure:**

```yaml
# data/toy/config.yaml
cosmology:
  Omega_m: 0.3
  sigma_8: 0.8
  h: 0.7

simulation:
  grid_shape: [128, 128]
  n_steps: 50
  dt: 0.05

observation:
  n_galaxies: 1000
  noise_level: 0.1
  survey_mask: rectangular

inference:
  optimizer: adam
  learning_rate: 0.001
  n_iterations: 1000
```

**Loading:**

```python
import yaml

def load_config(filepath):
    with open(filepath) as f:
        config = yaml.safe_load(f)
    return config

config = load_config('data/toy/config.yaml')
```

**Design Notes:**
- Config files for reproducibility
- Each experiment has its own config
- Scripts read config, notebooks can override
- Config schema validated at runtime (optional)

---

## Testing Architecture

Tests are organized to match module structure:

```
tests/
├── __init__.py
├── test_fields.py       # Test initial condition generation
├── test_pm.py           # Test PM evolution
├── test_observe.py      # Test observation operators
├── test_loss.py         # Test loss functions
├── test_inference.py    # Test optimizers/samplers
└── conftest.py          # Shared fixtures
```

**Testing Strategy:**

1. **Unit Tests** — Each function tested in isolation
   - Known input → expected output
   - Edge cases (empty arrays, zero values)
   - Gradient checks (finite differences)

2. **Integration Tests** — End-to-end pipelines
   - Forward model: initial conditions → evolved field
   - Inverse model: known truth → successful recovery
   - Reproducibility: same seed → same result

3. **Property Tests** — Mathematical properties
   - Power spectrum conservation
   - Symplectic integration (energy conservation)
   - Gradient correctness (finite difference comparison)

**Example Test Structure:**

```python
# tests/test_fields.py
import pytest
import jax.numpy as jnp
from diffcosmo import fields

def test_gaussian_field_shape():
    """Test that generated field has correct shape."""
    grid_shape = (64, 64)
    field = fields.generate_gaussian_field(grid_shape, seed=42)
    assert field.shape == grid_shape

def test_power_spectrum_match():
    """Test that field power spectrum matches target."""
    power_fn = lambda k: k**(-2)
    field = fields.generate_gaussian_field((128, 128), power_fn, seed=42)
    
    k, P_measured = fields.compute_power_spectrum(field)
    P_target = power_fn(k)
    
    # Should match within statistical fluctuations
    assert jnp.allclose(P_measured, P_target, rtol=0.1)

def test_field_gradient():
    """Test that gradients flow through field generation."""
    def loss_fn(seed_val):
        field = fields.generate_gaussian_field((32, 32), seed=int(seed_val))
        return jnp.sum(field**2)
    
    grad = jax.grad(loss_fn)(42.0)
    assert jnp.isfinite(grad)
```

---

## Extension Points

The architecture is designed for extensibility:

### 1. Adding a New Forward Model

Replace `pm.py` with alternative physics:

```python
# Create new file: src/diffcosmo/lpt.py
def evolve_lpt(initial_field, order=2):
    """Lagrangian Perturbation Theory evolution."""
    # Your implementation
    return evolved_field

# Use in loss function:
from diffcosmo import lpt
forward_model = lpt.evolve_lpt  # Instead of pm.evolve_pm
```

### 2. Adding a New Observation Operator

Extend `observe.py`:

```python
# Add to observe.py
def apply_rsd(galaxy_positions, velocity_field):
    """Apply redshift-space distortions."""
    # Your implementation
    return distorted_positions

# Compose with existing operators:
galaxies = sample_galaxies_poisson(field)
galaxies = apply_rsd(galaxies, velocity)
galaxies = apply_survey_mask(galaxies)
```

### 3. Adding a New Prior

Add to `loss.py`:

```python
def smoothness_prior(field, smoothness_scale):
    """Penalize high-frequency modes."""
    field_fft = jnp.fft.fftn(field)
    k = compute_k_values(field.shape)
    penalty = jnp.sum((k > smoothness_scale) * jnp.abs(field_fft)**2)
    return penalty

# Use in total loss:
def total_loss(theta):
    likelihood_term = gaussian_likelihood(...)
    ps_prior = power_spectrum_prior(...)
    smooth_prior = smoothness_prior(theta)
    return likelihood_term + ps_prior + smooth_prior
```

### 4. Adding a New Optimizer

Add to `inference.py`:

```python
def optimize_custom(loss_fn, initial_params, **kwargs):
    """Custom optimization algorithm."""
    params = initial_params
    for step in range(n_steps):
        loss, grad = jax.value_and_grad(loss_fn)(params)
        # Your optimization logic
        params = update_rule(params, grad)
    return params

# Use like any other optimizer:
result = optimize_custom(loss_fn, theta_init)
```

---

## Performance Considerations

### Memory Management

**Problem:** Large 3D grids (512³ cells) × many timesteps = huge memory

**Solutions implemented:**

1. **Gradient Checkpointing** (in `pm.py`)
   - Store states at intervals, recompute in between
   - Configurable: 'none', 'uniform', 'logarithmic'

2. **In-place Operations Where Safe**
   - Particle updates can be in-place (not backpropped through)
   - Grid operations make new arrays (needed for autodiff)

3. **Mixed Precision** (future)
   - Store in float32, compute in float64
   - Trade accuracy for 2× memory reduction

### Computation Speed

**Bottlenecks:**

1. **FFTs** (40-60% of time)
   - Use FFTW backend via JAX
   - Batch FFTs when possible

2. **Particle-Grid Assignment** (20-30%)
   - Vectorized with JAX
   - GPU-accelerated automatically

3. **Gradient Accumulation** (10-20%)
   - Inherent cost of autodiff
   - Checkpointing adds ~2× overhead

**Optimizations:**

- `@jax.jit` decorators on hot paths
- `vmap` for batching over multiple seeds
- `pmap` for multi-GPU (future)

### Scaling Strategy

**Single GPU:** 128³ to 256³ cells
**Multi-GPU:** 512³+ via domain decomposition (future work)

---

## Code Style and Conventions

### Naming Conventions

- **Functions:** `snake_case` (e.g., `evolve_pm`, `compute_loss`)
- **Classes:** `PascalCase` (e.g., `FlowSampler`, rare in this codebase)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_N_STEPS`)
- **Private functions:** prefix with `_` (e.g., `_internal_helper`)

### Type Hints

Use type hints for public API:

```python
def evolve_pm(
    initial_field: jnp.ndarray,
    n_steps: int,
    cosmology: dict
) -> jnp.ndarray:
    """Evolve field forward."""
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def function_name(arg1, arg2):
    """Short one-line description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When input is invalid
    """
```

### JAX Best Practices

- Use `jnp` not `np` for arrays inside functions
- Avoid Python loops, use `vmap` or `scan`
- Functions must be pure (no side effects)
- Use `jit` for performance, but debug without it first

---

## Common Patterns

### Pattern 1: Value and Grad

```python
# Compute loss and gradient simultaneously
loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
(loss, aux), grad = loss_and_grad_fn(params)
```

### Pattern 2: Scan for Time Loops

```python
# Instead of Python for-loop:
def step_fn(carry, inputs):
    state = carry
    new_state = pm_step(state, inputs)
    return new_state, new_state

final_state, trajectory = jax.lax.scan(step_fn, initial_state, timesteps)
```

### Pattern 3: Checkpointing with Recomputation

```python
@jax.checkpoint  # Recompute this in backward pass
def expensive_forward(x):
    return evolve_pm(x, n_steps=100)
```

### Pattern 4: Batching with vmap

```python
# Vectorize over multiple seeds
def run_single(seed):
    field = generate_gaussian_field(grid_shape, seed=seed)
    return evolve_pm(field)

seeds = jnp.array([1, 2, 3, 4, 5])
results = jax.vmap(run_single)(seeds)  # Parallel batch
```

---

## Future Architecture Changes

### Planned Refactoring (Phase 2+)

1. **Add Configuration Classes**
   - Replace dicts with dataclasses
   - Type-checked configuration
   - Better IDE support

2. **Separate 2D and 3D Code Paths**
   - Currently mixed in same functions
   - Create `pm_2d.py` and `pm_3d.py`
   - Reduces conditionals

3. **Plugin System for Observation Models**
   - Register custom operators
   - Compose via config file
   - Example: `observations: [poisson, mask, rsd, noise]`

4. **Multi-GPU via pmap**
   - Spatial domain decomposition
   - Cross-GPU FFT communication
   - Scales to 512³+ grids

5. **Automatic API Documentation**
   - Generate from docstrings
   - Sphinx integration
   - Hosted on Read the Docs

---

## Summary

**Key Architectural Decisions:**

1. ✅ **Functional paradigm** enables JAX autodiff
2. ✅ **Modular structure** keeps concerns separated
3. ✅ **Pure functions** make testing/debugging easier
4. ✅ **Explicit data flow** from fields → PM → observe → loss
5. ✅ **Configurable checkpointing** balances memory vs. compute
6. ✅ **Extension points** at every layer

**For Contributors:**

- Start with `fields.py` or `observe.py` (simplest)
- Understand `pm.py` before optimizing
- Read `loss.py` to understand what's being optimized
- Study `inference.py` for sampling algorithms

**Next Steps:**

- See [`getting_started.md`](getting_started.md) for a code walkthrough
- See [`development.md`](development.md) for contribution workflow
- See [`overview.md`](overview.md) for scientific background

---

This architecture is a living document. As the codebase evolves, update this file to reflect major structural changes.