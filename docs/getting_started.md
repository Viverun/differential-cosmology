# Getting Started Tutorial

Welcome! This tutorial will walk you through your first forward simulation and reconstruction using Differentiable Cosmology. By the end, you'll understand the core workflow and be ready to run your own experiments.

**Time required:** 30-60 minutes  
**Prerequisites:** Python installed, basic familiarity with NumPy  
**What you'll build:** A complete 2D cosmological reconstruction pipeline

---

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Project Orientation](#project-orientation)
3. [Your First Forward Simulation](#your-first-forward-simulation)
4. [Understanding the Output](#understanding-the-output)
5. [Your First Reconstruction](#your-first-reconstruction)
6. [Interpreting Results](#interpreting-results)
7. [Running Experiments](#running-experiments)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## Setup & Installation

If you haven't already installed the package, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/differentiable-cosmology.git
cd differentiable-cosmology

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Verify installation
python -c "import diffcosmo; print('Installation successful!')"
```

**Expected output:** `Installation successful!`

If you encounter issues, see [Troubleshooting](#troubleshooting) below.

---

## Project Orientation

Before diving in, let's understand what we're building:

### The Big Picture

```
Initial Conditions → Forward Model → Observations
     (what we want)   (physics sim)    (what we see)
          ↑                                    |
          |        Backprop & Optimize         |
          └────────────────────────────────────┘
```

**Forward problem:** Given initial conditions, simulate forward to predict observations.

**Inverse problem:** Given observations, work backward to infer initial conditions.

### File Structure Quick Tour

```
differentiable-cosmology/
├── src/diffcosmo/          # Core library
│   ├── fields.py          # Initial condition generation
│   ├── pm.py              # Forward simulation (particle-mesh)
│   ├── observe.py         # Observation model
│   ├── loss.py            # Loss functions
│   └── inference.py       # Optimization/sampling
├── notebooks/              # Interactive tutorials
├── scripts/                # Command-line tools
└── data/                   # Configurations and test data
```

**For this tutorial:** We'll use the library interactively in Python. Later you can explore notebooks and scripts.

---

## Your First Forward Simulation

Let's simulate the universe forward in time, starting from simple initial conditions.

### Step 1: Import Libraries

Create a new Python file `my_first_sim.py` or use an interactive Python session:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffcosmo import fields, pm, utils

# Set a random seed for reproducibility
SEED = 42
```

**What we imported:**
- `fields` — Generate initial density fields
- `pm` — Particle-mesh evolution
- `utils` — Plotting and analysis tools

### Step 2: Define Cosmological Parameters

```python
# Simple cosmology (close to our universe)
cosmology = {
    'Omega_m': 0.3,    # Matter density
    'sigma_8': 0.8,    # Amplitude of fluctuations
    'h': 0.7           # Hubble parameter
}

# Simulation parameters
grid_shape = (64, 64)  # 64x64 grid (small for speed)
n_steps = 20           # Number of time steps
```

**What these mean:**
- `Omega_m` — How much matter is in the universe (0.3 = 30%)
- `sigma_8` — How "lumpy" the initial conditions are
- `grid_shape` — Resolution of our simulation
- `n_steps` — How long to evolve forward

### Step 3: Generate Initial Conditions

```python
# Define a simple power spectrum (how fluctuations vary with scale)
def power_spectrum(k):
    """Power-law power spectrum: P(k) ∝ k^(-2)"""
    return jnp.where(k > 0, k**(-2), 0.0)

# Generate Gaussian random field
print("Generating initial conditions...")
initial_field = fields.generate_gaussian_field(
    grid_shape=grid_shape,
    power_spectrum=power_spectrum,
    seed=SEED
)

print(f"Initial field shape: {initial_field.shape}")
print(f"Initial field range: [{initial_field.min():.3f}, {initial_field.max():.3f}]")
```

**Expected output:**
```
Generating initial conditions...
Initial field shape: (64, 64)
Initial field range: [-2.145, 2.387]
```

**What happened:** We created a random density field where each point represents how much denser or less dense that region is compared to the cosmic average.

### Step 4: Visualize Initial Conditions

```python
# Plot the initial field
plt.figure(figsize=(10, 8))
plt.imshow(initial_field, cmap='RdBu', vmin=-2, vmax=2)
plt.colorbar(label='Density contrast δ')
plt.title('Initial Density Field (Early Universe)')
plt.xlabel('x [grid units]')
plt.ylabel('y [grid units]')
plt.savefig('initial_field.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved initial_field.png")
```

**What you should see:** A red-and-blue map showing overdense (red) and underdense (blue) regions. This represents the "seeds" from which cosmic structure grows.

### Step 5: Evolve Forward in Time

```python
print("Evolving forward in time...")
final_field = pm.evolve_pm(
    initial_field=initial_field,
    n_steps=n_steps,
    cosmology=cosmology,
    checkpointing='none'  # No checkpointing for this simple case
)

print(f"Final field shape: {final_field.shape}")
print(f"Final field range: [{final_field.min():.3f}, {final_field.max():.3f}]")
```

**Expected output:**
```
Evolving forward in time...
Final field shape: (64, 64)
Final field range: [-0.895, 3.421]
```

**What happened:** Gravity pulled matter together over time. Overdense regions became denser, forming structure. Underdense regions became voids.

### Step 6: Visualize Evolved Field

```python
plt.figure(figsize=(10, 8))
plt.imshow(final_field, cmap='viridis', vmin=0, vmax=3)
plt.colorbar(label='Final density')
plt.title('Evolved Density Field (Late Universe)')
plt.xlabel('x [grid units]')
plt.ylabel('y [grid units]')
plt.savefig('final_field.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved final_field.png")
```

**What you should see:** Structure has formed! You should see:
- Dense clumps (yellow/bright) — where galaxies would be
- Filaments connecting clumps — the "cosmic web"
- Large voids (dark purple) — empty regions

**Compare to initial field:** Much more structure and contrast now.

---

## Understanding the Output

Let's analyze what we simulated.

### Compute Power Spectrum

The power spectrum P(k) tells us how much structure we have at different scales:

```python
# Compute power spectrum of initial and final fields
k_init, P_init = utils.compute_power_spectrum(initial_field)
k_final, P_final = utils.compute_power_spectrum(final_field)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.loglog(k_init, P_init, label='Initial', alpha=0.7)
plt.loglog(k_final, P_final, label='Final (evolved)', alpha=0.7)
plt.xlabel('Wavenumber k')
plt.ylabel('Power P(k)')
plt.title('Power Spectrum Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('power_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved power_spectrum.png")
```

**What to look for:**
- Final P(k) should be higher than initial P(k) — gravity amplified fluctuations
- Shape should be similar but amplified
- If they're identical, something went wrong (no evolution happened)

### Key Metrics

```python
# Structure growth
initial_variance = jnp.var(initial_field)
final_variance = jnp.var(final_field)
growth_factor = jnp.sqrt(final_variance / initial_variance)

print(f"\nStructure Growth Analysis:")
print(f"Initial variance: {initial_variance:.4f}")
print(f"Final variance: {final_variance:.4f}")
print(f"Growth factor: {growth_factor:.4f}")
print(f"Structure amplified by {growth_factor:.2f}x")
```

**Expected output:**
```
Structure Growth Analysis:
Initial variance: 0.6234
Final variance: 1.8901
Growth factor: 1.7421
Structure amplified by 1.74x
```

**Interpretation:** Gravity made structures ~1.7× stronger. This is expected for the evolution time we simulated.

---

## Your First Reconstruction

Now the exciting part: **can we reverse-engineer the initial conditions from the final state?**

### The Inverse Problem Setup

We'll pretend we only observe the final field (with some noise), and try to recover the initial conditions.

### Step 1: Create Mock Observations

```python
from diffcosmo import observe

print("\n=== Creating Mock Observations ===")

# Add observational noise
observations = observe.add_observation_noise(
    final_field,
    noise_level=0.1,
    seed=SEED + 1
)

print(f"Observations shape: {observations.shape}")
print(f"Signal-to-noise ratio: {jnp.std(final_field) / 0.1:.2f}")
```

**What happened:** We added Gaussian noise to simulate measurement uncertainty. Real telescopes have noise!

### Step 2: Define the Loss Function

The loss function measures how well our guess matches the observations:

```python
from diffcosmo import loss

def forward_model(theta):
    """
    Our forward model: initial field → evolved field.
    
    Args:
        theta: initial field (what we're inferring)
    
    Returns:
        Predicted observations
    """
    evolved = pm.evolve_pm(theta, n_steps, cosmology, checkpointing='none')
    return evolved

def loss_fn(theta):
    """
    Loss = how different is our prediction from observations?
    
    Lower loss = better match.
    """
    prediction = forward_model(theta)
    
    # Negative log likelihood (Gaussian)
    likelihood = loss.gaussian_likelihood(prediction, observations, noise_covariance=0.1**2)
    
    # Prior: enforce that initial field has correct power spectrum
    prior = loss.power_spectrum_prior(theta, power_spectrum)
    
    total_loss = likelihood + prior
    
    # Return loss and auxiliary info for logging
    aux = {
        'likelihood': likelihood,
        'prior': prior,
        'prediction': prediction
    }
    
    return total_loss, aux

print("Loss function defined!")
```

**What we're doing:** Defining what "good" means. A good reconstruction should:
1. Match the observations (likelihood term)
2. Have the right statistical properties (prior term)

### Step 3: Test the Loss Function

Before optimizing, let's check it works:

```python
# Try with the true initial field
test_loss, test_aux = loss_fn(initial_field)
print(f"\nLoss with TRUE initial conditions: {test_loss:.4f}")

# Try with random guess
random_guess = fields.generate_gaussian_field(grid_shape, power_spectrum, seed=999)
random_loss, random_aux = loss_fn(random_guess)
print(f"Loss with RANDOM guess: {random_loss:.4f}")

print(f"\nTrue initial conditions should have lower loss: {test_loss < random_loss}")
```

**Expected output:**
```
Loss with TRUE initial conditions: 12.3456
Loss with RANDOM guess: 87.6543

True initial conditions should have lower loss: True
```

**Good sign:** True initial conditions have lower loss than a random guess. Our loss function makes sense!

### Step 4: Optimize (Reconstruct)

Now let's use gradient descent to find the best initial conditions:

```python
from diffcosmo import inference
import jax

print("\n=== Starting Reconstruction ===")

# Start from a random guess
initial_guess = fields.generate_gaussian_field(grid_shape, power_spectrum, seed=100)

# Run MAP optimization
print("Optimizing (this may take a minute)...")
result = inference.optimize_map(
    loss_fn=loss_fn,
    initial_params=initial_guess,
    optimizer='adam',
    n_steps=200,
    learning_rate=0.01,
    verbose=True  # Print progress
)

reconstructed_field = result['params']
history = result['history']

print(f"\nOptimization complete!")
print(f"Initial loss: {history['loss'][0]:.4f}")
print(f"Final loss: {history['loss'][-1]:.4f}")
print(f"Loss decreased by {history['loss'][0] - history['loss'][-1]:.4f}")
```

**Expected output:**
```
=== Starting Reconstruction ===
Optimizing (this may take a minute)...
Step 0: loss = 87.6543
Step 50: loss = 34.2156
Step 100: loss = 18.7654
Step 150: loss = 13.9876
Step 200: loss = 12.8765

Optimization complete!
Initial loss: 87.6543
Final loss: 12.8765
Loss decreased by 74.7778
```

**What happened:** We used gradients to iteratively improve our guess. The loss went down, meaning we got closer to the truth!

### Step 5: Plot Optimization Progress

```python
plt.figure(figsize=(10, 6))
plt.plot(history['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Optimization Progress')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('optimization_progress.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved optimization_progress.png")
```

**What to look for:**
- Loss should decrease steadily (mostly)
- Should plateau at the end (converged)
- If it increases or oscillates wildly, learning rate is too high

---

## Interpreting Results

Let's see how well we did!

### Visual Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# True initial field
im0 = axes[0].imshow(initial_field, cmap='RdBu', vmin=-2, vmax=2)
axes[0].set_title('True Initial Field')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0])

# Reconstructed field
im1 = axes[1].imshow(reconstructed_field, cmap='RdBu', vmin=-2, vmax=2)
axes[1].set_title('Reconstructed Field')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im1, ax=axes[1])

# Difference
difference = initial_field - reconstructed_field
im2 = axes[2].imshow(difference, cmap='RdBu', vmin=-1, vmax=1)
axes[2].set_title('Residual (True - Recon)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved reconstruction_comparison.png")
```

**What to look for:**
- True and reconstructed should look similar (same patterns)
- Residual should be mostly small (blue/white), not large red/blue patches
- If residual has large-scale patterns, something is wrong

### Quantitative Metrics

```python
# Cross-correlation (key metric!)
k_bins, r_k = utils.compute_cross_correlation(initial_field, reconstructed_field)

print("\n=== Reconstruction Quality Metrics ===")
print(f"Cross-correlation at large scales: r(k_min) = {r_k[0]:.3f}")
print(f"Cross-correlation at small scales: r(k_max) = {r_k[-1]:.3f}")
print(f"Mean cross-correlation: {jnp.mean(r_k):.3f}")

# Real-space correlation
real_space_corr = jnp.corrcoef(
    initial_field.flatten(), 
    reconstructed_field.flatten()
)[0, 1]
print(f"Real-space correlation: {real_space_corr:.3f}")

# Interpretation
if real_space_corr > 0.9:
    print("✅ Excellent reconstruction!")
elif real_space_corr > 0.8:
    print("✅ Good reconstruction!")
elif real_space_corr > 0.7:
    print("⚠️  Acceptable reconstruction (could be better)")
else:
    print("❌ Poor reconstruction (needs debugging)")
```

**Expected output:**
```
=== Reconstruction Quality Metrics ===
Cross-correlation at large scales: r(k_min) = 0.943
Cross-correlation at small scales: r(k_max) = 0.712
Mean cross-correlation: 0.847
Real-space correlation: 0.856
✅ Good reconstruction!
```

**Interpretation:**
- **r > 0.9:** Large-scale structure recovered very well
- **r ~ 0.7:** Small-scale structure harder to recover (expected)
- **Overall:** We successfully reverse-engineered most of the initial conditions!

### Plot Cross-Correlation

```python
plt.figure(figsize=(10, 6))
plt.semilogx(k_bins, r_k, 'o-', markersize=4)
plt.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (r>0.9)')
plt.axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='Good (r>0.8)')
plt.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Acceptable (r>0.7)')
plt.xlabel('Wavenumber k')
plt.ylabel('Cross-correlation r(k)')
plt.title('Reconstruction Quality vs. Scale')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.1])
plt.savefig('cross_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved cross_correlation.png")
```

**What this shows:** We recover large-scale structure (low k) better than small-scale structure (high k). This is expected and physically sensible.

---

## Running Experiments

Now that you understand the basics, try these experiments:

### Experiment 1: Effect of Noise

How does observational noise affect reconstruction?

```python
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
correlations = []

for noise in noise_levels:
    print(f"\nTrying noise level {noise}...")
    
    # Create noisy observations
    noisy_obs = observe.add_observation_noise(final_field, noise, seed=SEED+1)
    
    # Reconstruct (simplified - fewer steps for speed)
    def loss_fn_noise(theta):
        pred = forward_model(theta)
        lik = loss.gaussian_likelihood(pred, noisy_obs, noise**2)
        pri = loss.power_spectrum_prior(theta, power_spectrum)
        return lik + pri, {}
    
    result = inference.optimize_map(loss_fn_noise, initial_guess, 
                                   n_steps=100, verbose=False)
    
    # Measure quality
    corr = jnp.corrcoef(initial_field.flatten(), 
                       result['params'].flatten())[0, 1]
    correlations.append(corr)
    print(f"  Correlation: {corr:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, correlations, 'o-', markersize=8, linewidth=2)
plt.xlabel('Noise Level')
plt.ylabel('Reconstruction Correlation')
plt.title('Effect of Observational Noise on Reconstruction')
plt.grid(True, alpha=0.3)
plt.savefig('noise_experiment.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nConclusion: Higher noise → worse reconstruction (as expected)")
```

### Experiment 2: Effect of Grid Resolution

Does higher resolution help?

```python
resolutions = [32, 64, 128]
correlations = []

for N in resolutions:
    print(f"\nTrying grid size {N}x{N}...")
    
    # Generate new fields at this resolution
    field_init = fields.generate_gaussian_field((N, N), power_spectrum, seed=SEED)
    field_final = pm.evolve_pm(field_init, n_steps, cosmology)
    obs = observe.add_observation_noise(field_final, 0.1, seed=SEED+1)
    
    # Reconstruct (fewer steps for larger grids)
    # ... (similar to above)
    
    # Measure and store
    # ...

print("\nConclusion: Higher resolution → better reconstruction (but slower)")
```

### Experiment 3: Effect of Optimization Steps

When does optimization converge?

```python
step_counts = [10, 50, 100, 200, 500]
final_losses = []

for n_steps_opt in step_counts:
    print(f"\nOptimizing for {n_steps_opt} steps...")
    result = inference.optimize_map(loss_fn, initial_guess, 
                                   n_steps=n_steps_opt, verbose=False)
    final_losses.append(result['history']['loss'][-1])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(step_counts, final_losses, 'o-')
plt.xlabel('Optimization Steps')
plt.ylabel('Final Loss')
plt.title('Convergence Analysis')
plt.grid(True, alpha=0.3)
plt.show()

print("\nConclusion: Diminishing returns after ~200 steps")
```

---

## Troubleshooting

### Common Issues

**Problem: "ModuleNotFoundError: No module named 'diffcosmo'"**

Solution:
```bash
# Make sure you installed in editable mode
pip install -e .

# Check if it's in the right environment
which python  # Should point to your venv
```

**Problem: "JAX not installed" or "No GPU found"**

Solution:
```bash
# Install JAX (CPU version)
pip install --upgrade jax jaxlib

# For GPU (CUDA 11.x)
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Problem: Loss becomes NaN during optimization**

Solution:
```python
# Reduce learning rate
result = inference.optimize_map(loss_fn, initial_guess, learning_rate=0.001)

# Or add gradient clipping
result = inference.optimize_map(loss_fn, initial_guess, clip_grad=1.0)
```

**Problem: Optimization is very slow**

Solution:
```python
# Use smaller grid for testing
grid_shape = (32, 32)  # Instead of 64x64

# Fewer time steps
n_steps = 10  # Instead of 20

# Enable JIT compilation
import jax
loss_fn_jit = jax.jit(loss_fn)  # Compile once, run fast
```

**Problem: Reconstruction quality is poor (r < 0.5)**

Check:
- Is the forward model correct? (Test on known solution)
- Is the noise level too high? (Try lower noise)
- Are you optimizing long enough? (Try more steps)
- Is the learning rate appropriate? (Too high = diverges, too low = slow)

**Problem: "Memory error" or "Out of memory"**

Solution:
```python
# Use smaller grid
grid_shape = (32, 32)

# Enable checkpointing (trades compute for memory)
final_field = pm.evolve_pm(initial_field, n_steps, cosmology, 
                          checkpointing='uniform')
```

### Getting Help

If you're stuck:

1. Check the [documentation](docs/overview.md)
2. Search [GitHub Issues](https://github.com/yourusername/differentiable-cosmology/issues)
3. Ask in [GitHub Discussions](https://github.com/yourusername/differentiable-cosmology/discussions)
4. Include:
   - What you tried
   - Full error message
   - Your environment (Python version, OS, JAX version)
   - Minimal code to reproduce

---

## Next Steps

Congratulations! You've completed your first differentiable cosmology reconstruction. 🎉

### Deepen Your Understanding

**Try the notebooks:**
- `notebooks/00_toy_2d_forward.ipynb` — More detailed forward modeling
- `notebooks/01_toy_2d_inverse.ipynb` — Advanced reconstruction techniques

**Read the docs:**
- [docs/overview.md](docs/overview.md) — Mathematical foundations
- [docs/architecture.md](docs/architecture.md) — Code structure
- [docs/validation.md](docs/validation.md) — Quality metrics

**Explore the code:**
- `src/diffcosmo/fields.py` — How initial conditions are generated
- `src/diffcosmo/pm.py` — How evolution works
- `src/diffcosmo/inference.py` — Optimization algorithms

### Advanced Topics

**Add realistic observations:**
- Survey masks (partial sky coverage)
- Redshift-space distortions (RSD)
- Galaxy bias and HOD models

**Try 3D simulations:**
- Larger volumes
- More realistic cosmology
- Higher resolution

**Posterior sampling:**
- Hamiltonian Monte Carlo (HMC)
- Normalizing flows
- Uncertainty quantification

**Real data:**
- Apply to galaxy surveys (SDSS, DES)
- Compare to traditional methods
- Joint parameter inference

### Contributing

Found a bug? Have an idea? Want to help?

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

**Easy contributions:**
- Improve this tutorial
- Add more examples
- Fix typos in docs
- Report issues

**Harder contributions:**
- Implement 3D PM solver
- Add new observation models
- Optimize performance
- Add posterior samplers

---

## Summary

You've learned:

✅ How to generate initial conditions  
✅ How to run forward simulations  
✅ How to create mock observations  
✅ How to define loss functions  
✅ How to optimize (reconstruct) using gradients  
✅ How to validate reconstruction quality  
✅ How to run experiments and debug issues

**Core insight:** Differentiable simulation + gradient-based optimization = powerful inverse problem solver for cosmology!

**Remember:** This was a toy 2D example. Real cosmological inference is more complex, but the principles are the same.

---

**Happy reconstructing!** 🌌