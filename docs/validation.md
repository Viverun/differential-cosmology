# Validation & Success Criteria

This document defines how we validate the Differentiable Cosmology codebase and measure the quality of our reconstructions. It's essential reading for contributors implementing new features or running scientific experiments.

---

## Overview: What We're Validating

Differentiable cosmology has two layers that need validation:

**1. Software Correctness** — Does the code do what it's supposed to?
- Forward model matches known physics
- Gradients are computed correctly
- Numerical integration is stable
- Code is deterministic and reproducible

**2. Scientific Quality** — Does the inference work well?
- Reconstructed fields match ground truth
- Posterior estimates are well-calibrated
- Method scales to realistic problems
- Results are scientifically meaningful

Both are critical. Correct code that produces bad science is useless. Good science from buggy code is unreliable.

---

## Core Validation Metrics

### 1. Cross-Correlation Coefficient

**What it measures:** How well the reconstructed initial field matches the true initial field.

**Definition:**
```
r(k) = <δ_true(k) · δ_recon*(k)> / sqrt(<|δ_true(k)|²> · <|δ_recon(k)|²>)
```

where `< >` denotes averaging over Fourier modes at wavenumber `k`.

**Success criteria:**
- **Excellent:** r(k) > 0.95 for k < k_nyquist/2
- **Good:** r(k) > 0.90 for k < k_nyquist/2  
- **Acceptable:** r(k) > 0.80 for k < k_nyquist/2
- **Poor:** r(k) < 0.80 (needs investigation)

**Interpretation:**
- r = 1.0: Perfect reconstruction
- r = 0.9: Captures 90% of the signal
- r = 0.0: No correlation (random guess)
- r < 0.0: Anti-correlated (something is wrong!)

**Why it matters:** Directly measures field-level reconstruction quality. Unlike power spectrum (which only measures amplitude), this measures phase information too.

**Compute with:**
```python
from diffcosmo.utils import compute_cross_correlation

k_bins, r_k = compute_cross_correlation(field_true, field_recon)
```

### 2. Power Spectrum Recovery

**What it measures:** Whether the reconstructed field has the correct statistical properties.

**Definition:**
```
Fractional error = |P_recon(k) - P_true(k)| / P_true(k)
```

**Success criteria:**
- **Excellent:** <5% error for k < k_nyquist/2
- **Good:** <10% error for k < k_nyquist/2
- **Acceptable:** <20% error for k < k_nyquist/2
- **Poor:** >20% error (check for systematic bias)

**Why it matters:** Power spectrum measures amplitude correctness. Even if phases are wrong, we should recover the right P(k).

**Compute with:**
```python
from diffcosmo.utils import compute_power_spectrum, bin_power_spectrum

k_true, P_true = compute_power_spectrum(field_true)
k_recon, P_recon = compute_power_spectrum(field_recon)

# Bin for cleaner comparison
k_binned, P_true_binned = bin_power_spectrum(k_true, P_true, n_bins=20)
_, P_recon_binned = bin_power_spectrum(k_recon, P_recon, n_bins=20)

fractional_error = np.abs(P_recon_binned - P_true_binned) / P_true_binned
```

### 3. Real-Space Correlation

**What it measures:** How well spatial structure is recovered.

**Definition:**
```
Real-space r = Pearson correlation(δ_true, δ_recon)
```

**Success criteria:**
- **Excellent:** r > 0.90
- **Good:** r > 0.80
- **Acceptable:** r > 0.70
- **Poor:** r < 0.70

**Why it matters:** Complements Fourier-space metrics. Easy to visualize and interpret.

**Compute with:**
```python
import numpy as np

r_real = np.corrcoef(field_true.flatten(), field_recon.flatten())[0, 1]
```

### 4. Visual Inspection

**What it measures:** Qualitative assessment of reconstruction.

**What to check:**
- Do overdensities align spatially?
- Are filamentary structures preserved?
- Are there obvious artifacts or noise?
- Does the reconstruction "look like" the truth?

**Success criteria:**
Visual inspection should pass the "eyeball test" — an expert should say "yes, that looks right."

**Generate with:**
```python
from diffcosmo.utils import plot_field_comparison

plot_field_comparison(field_true, field_recon, 
                     title="True vs. Reconstructed")
```

---

## Optimization Validation

### 5. Loss Convergence

**What it measures:** Whether optimization is working properly.

**Success criteria:**
- Loss decreases monotonically (or mostly so)
- Loss plateaus at reasonable value
- No sudden jumps or NaN values
- Final loss is significantly better than initial loss

**Warning signs:**
- Loss increases over time → learning rate too high
- Loss oscillates wildly → unstable optimization
- Loss becomes NaN → numerical overflow, check gradients
- Loss plateaus immediately → stuck in local minimum or gradient is zero

**Compute with:**
```python
history = optimize_map(loss_fn, initial_params, ...)

plt.plot(history['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Optimization Trajectory')
```

### 6. Gradient Sanity Checks

**What it measures:** Whether gradients are computed correctly.

**Tests:**

**a) Finite Difference Comparison**
```python
def check_gradients(loss_fn, params, epsilon=1e-5):
    """Compare autodiff gradients to finite differences."""
    analytic_grad = jax.grad(loss_fn)(params)
    
    numerical_grad = np.zeros_like(params)
    for i in range(params.size):
        params_plus = params.copy()
        params_plus.flat[i] += epsilon
        params_minus = params.copy()
        params_minus.flat[i] -= epsilon
        
        numerical_grad.flat[i] = (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * epsilon)
    
    relative_error = np.abs(analytic_grad - numerical_grad) / (np.abs(numerical_grad) + 1e-10)
    
    return relative_error.max() < 1e-3  # Should be < 0.1%
```

**b) Gradient Norm Tracking**
```python
# Gradients should not explode or vanish
grad_norms = [np.linalg.norm(grad) for grad in history['gradients']]

assert all(np.isfinite(grad_norms))
assert max(grad_norms) < 1e6  # No explosion
assert min(grad_norms) > 1e-10  # No vanishing
```

### 7. Reproducibility

**What it measures:** Whether results are deterministic.

**Test:**
```python
# Run twice with same seed
result1 = optimize_map(loss_fn, initial_params, seed=42)
result2 = optimize_map(loss_fn, initial_params, seed=42)

assert np.allclose(result1, result2, rtol=1e-10)

# Run with different seeds should give different results
result3 = optimize_map(loss_fn, initial_params, seed=43)
assert not np.allclose(result1, result3)
```

**Success criteria:** Identical inputs → identical outputs (within floating point precision).

---

## Forward Model Validation

### 8. Energy Conservation (Symplectic Integration)

**What it measures:** Whether PM integration conserves energy.

**Test:**
```python
def test_energy_conservation():
    particles = initialize_particles(initial_field)
    
    energies = []
    for step in range(n_steps):
        particles = pm_step(particles, ...)
        E = compute_energy(particles)  # Kinetic + potential
        energies.append(E)
    
    # Energy should be approximately constant
    E_variation = (max(energies) - min(energies)) / energies[0]
    assert E_variation < 0.01  # <1% variation
```

**Success criteria:** Total energy (kinetic + potential) should remain constant to within ~1% for symplectic integrators.

### 9. Power Spectrum Evolution

**What it measures:** Whether evolution produces correct P(k) growth.

**Test:**
```python
def test_power_spectrum_evolution():
    # Start with linear P(k)
    P_linear = lambda k: k**(-2)
    initial_field = generate_gaussian_field(grid_shape, P_linear, seed=42)
    
    # Evolve forward
    final_field = evolve_pm(initial_field, n_steps=50)
    
    # Measure final P(k)
    k, P_final = compute_power_spectrum(final_field)
    
    # Compare to theoretical prediction (e.g., Zel'dovich approximation)
    P_theory = zel'dovich_power_spectrum(k, growth_factor)
    
    # Should match at large scales (k < k_nl)
    k_linear = k < k_nonlinear
    assert np.allclose(P_final[k_linear], P_theory[k_linear], rtol=0.1)
```

**Success criteria:** Evolved P(k) should match theoretical predictions or high-fidelity N-body simulations at large scales.

### 10. Grid Resolution Convergence

**What it measures:** Whether results converge as resolution increases.

**Test:**
```python
def test_resolution_convergence():
    resolutions = [64, 128, 256]
    results = []
    
    for N in resolutions:
        field = evolve_pm(initial_field, grid_shape=(N, N))
        results.append(field)
    
    # Compare 128 vs 256 (should be closer than 64 vs 128)
    diff_low = np.abs(results[0] - results[1]).mean()
    diff_high = np.abs(results[1] - results[2]).mean()
    
    assert diff_high < diff_low  # Should converge
```

**Success criteria:** Differences decrease as resolution increases (second-order convergence for PM).

---

## Posterior Inference Validation

### 11. Posterior Coverage (Calibration)

**What it measures:** Whether uncertainty estimates are accurate.

**Test (for sampling methods):**
```python
def test_posterior_coverage():
    # Generate synthetic data
    true_params = generate_true_parameters()
    observations = forward_model(true_params) + noise
    
    # Sample posterior
    samples = sample_posterior(observations, n_samples=1000)
    
    # Check if true params fall within credible intervals
    for i in range(n_params):
        ci_low, ci_high = np.percentile(samples[:, i], [16, 84])  # 68% CI
        coverage[i] = ci_low < true_params[i] < ci_high
    
    # 68% of parameters should be in 68% CI (calibrated)
    assert 0.6 < coverage.mean() < 0.75
```

**Success criteria:** 68% credible intervals should contain the true value ~68% of the time across many experiments.

### 12. Rank Statistics (SBC)

**What it measures:** Whether posterior sampling is unbiased.

**Method:** Simulation-Based Calibration (SBC)

**Test:**
```python
def test_sbc():
    ranks = []
    
    for trial in range(n_trials):
        # Sample true params from prior
        true_params = sample_prior()
        
        # Generate data
        data = forward_model(true_params) + noise
        
        # Sample posterior
        posterior_samples = sample_posterior(data, n_samples=100)
        
        # Compute rank of true value
        rank = (posterior_samples < true_params).sum()
        ranks.append(rank)
    
    # Ranks should be uniformly distributed
    # (Use chi-square test or visual inspection)
    plt.hist(ranks, bins=20)
    # Should look flat
```

**Success criteria:** Histogram of ranks should be approximately uniform (flat).

### 13. Posterior Contraction

**What it measures:** Whether data updates the prior.

**Test:**
```python
def test_posterior_contraction():
    prior_samples = sample_prior(n=1000)
    posterior_samples = sample_posterior(data, n=1000)
    
    prior_std = prior_samples.std(axis=0)
    posterior_std = posterior_samples.std(axis=0)
    
    # Posterior should be tighter than prior
    assert (posterior_std < prior_std).all()
    
    # Quantify information gain
    info_gain = -np.log(posterior_std / prior_std)
    assert (info_gain > 0).all()  # All parameters learned something
```

**Success criteria:** Posterior uncertainty < prior uncertainty for all parameters.

---

## Integration Tests

### 14. End-to-End Reconstruction

**What it measures:** Full pipeline from truth to reconstruction.

**Test:**
```python
def test_end_to_end_reconstruction():
    # 1. Generate ground truth
    true_field = generate_gaussian_field(grid_shape, power_spectrum, seed=42)
    
    # 2. Forward model (simulate observations)
    evolved_field = evolve_pm(true_field, n_steps=50)
    observations = apply_observations(evolved_field, noise_level=0.1)
    
    # 3. Inverse problem (reconstruct)
    def loss_fn(params):
        predicted = forward_model(params)
        return gaussian_likelihood(predicted, observations)
    
    recon_field = optimize_map(loss_fn, initial_guess, n_iterations=1000)
    
    # 4. Validate reconstruction
    r = compute_cross_correlation(true_field, recon_field)
    assert r > 0.8  # Should recover most of the signal
```

**Success criteria:** High cross-correlation (r > 0.8) in controlled scenario.

### 15. Known Solution Recovery

**What it measures:** Can we recover a known answer?

**Test cases:**

**a) Zero field:**
```python
true_field = np.zeros(grid_shape)
observations = forward_model(true_field)  # Should be zeros + noise
recon_field = reconstruct(observations)
assert np.allclose(recon_field, 0, atol=noise_level)
```

**b) Delta function:**
```python
true_field = np.zeros(grid_shape)
true_field[N//2, N//2] = 1.0  # Single spike
# Should reconstruct spike location
```

**c) Sinusoidal field:**
```python
x, y = np.meshgrid(np.arange(N), np.arange(N))
true_field = np.sin(2 * np.pi * x / N)
# Should reconstruct sine wave
```

**Success criteria:** Near-perfect reconstruction (r > 0.99) for simple analytical test cases.

---

## Performance Benchmarks

### 16. Runtime Scaling

**What it measures:** How runtime scales with problem size.

**Benchmark:**
```python
def benchmark_scaling():
    grid_sizes = [32, 64, 128, 256]
    runtimes = []
    
    for N in grid_sizes:
        start = time.time()
        field = evolve_pm(initial_field, grid_shape=(N, N, N), n_steps=50)
        runtimes.append(time.time() - start)
    
    # PM should scale as O(N^3 log N) due to FFTs
    # Plot log(runtime) vs log(N) → slope should be ~3
```

**Expected scaling:**
- Forward model: O(N³ log N) per timestep
- Full optimization: O(iterations × N³ log N)

**Success criteria:** Actual scaling matches theoretical prediction.

### 17. Memory Usage

**What it measures:** Peak memory consumption.

**Benchmark:**
```python
import tracemalloc

tracemalloc.start()
field = evolve_pm(initial_field, grid_shape=(256, 256, 256))
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Peak memory: {peak / 1e9:.2f} GB")
```

**Expected memory:**
- Without checkpointing: O(T × N³) where T = timesteps
- With checkpointing: O(√T × N³)

**Success criteria:** Memory usage within expected bounds for given checkpointing strategy.

---

## Quality Assurance Workflow

### Pre-Commit Checks

Before committing code, ensure:
- [ ] All unit tests pass (`pytest tests/`)
- [ ] Code is formatted (`black src/ tests/`)
- [ ] Type hints are correct (`mypy src/`)
- [ ] Docstrings are present
- [ ] No TODOs or FIXMEs left unaddressed

### Pull Request Validation

Before merging PRs, verify:
- [ ] CI tests pass
- [ ] New code has tests (aim for >80% coverage)
- [ ] Documentation updated if API changed
- [ ] Performance benchmarks run (if relevant)
- [ ] Integration test with full pipeline

### Release Validation

Before each release:
- [ ] All validation metrics meet success criteria
- [ ] Benchmark results documented
- [ ] Example notebooks run end-to-end
- [ ] Documentation reviewed and updated
- [ ] Known issues documented

---

## Validation Checklist by Phase

### Phase 1 (2D Toy)

**Minimum passing criteria:**
- ✅ Forward model energy conserved (<1% variation)
- ✅ Gradients match finite differences (<0.1% error)
- ✅ Reconstruction r > 0.8 on toy problem
- ✅ Power spectrum recovered (<10% error)
- ✅ Results reproducible (same seed → same output)
- ✅ End-to-end notebook runs without errors

### Phase 2 (3D + Observations)

**Additional requirements:**
- ✅ 3D evolution matches 2D results (on 2D slice)
- ✅ Observation model differentiable
- ✅ Reconstruction r > 0.8 on 3D mock survey
- ✅ Survey mask handled correctly
- ✅ Multiple observation realizations consistent

### Phase 3 (Posterior Sampling)

**Additional requirements:**
- ✅ Posterior coverage properly calibrated
- ✅ SBC rank statistics uniform
- ✅ Posterior contracts from prior
- ✅ MCMC diagnostics (R̂, ESS) acceptable
- ✅ Flow sampler produces valid samples

### Phase 4 (Scale & Realism)

**Additional requirements:**
- ✅ Scales to 512³+ on multi-GPU
- ✅ RSD implementation validated against theory
- ✅ Baryonic effects marginalized properly
- ✅ Real survey mask applied correctly

---

## Diagnostic Plots

Essential visualizations for validation:

**1. Field Comparison (2D slice)**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(field_true[N//2], cmap='RdBu')
axes[0].set_title('True Field')
axes[1].imshow(field_recon[N//2], cmap='RdBu')
axes[1].set_title('Reconstructed Field')
axes[2].imshow(field_true[N//2] - field_recon[N//2], cmap='RdBu')
axes[2].set_title('Residual')
```

**2. Power Spectrum Comparison**
```python
plt.loglog(k_true, P_true, label='True', alpha=0.7)
plt.loglog(k_recon, P_recon, label='Reconstructed', alpha=0.7)
plt.xlabel('k')
plt.ylabel('P(k)')
plt.legend()
```

**3. Cross-Correlation vs. k**
```python
plt.semilogx(k_bins, r_k)
plt.axhline(0.9, color='red', linestyle='--', label='Target')
plt.xlabel('k')
plt.ylabel('r(k)')
plt.legend()
```

**4. Loss Trajectory**
```python
plt.semilogy(history['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Optimization Convergence')
```

**5. Scatter Plot (real-space)**
```python
plt.scatter(field_true.flatten(), field_recon.flatten(), alpha=0.1, s=1)
plt.plot([-3, 3], [-3, 3], 'r--', label='Perfect')
plt.xlabel('True δ')
plt.ylabel('Reconstructed δ')
plt.legend()
```

---

## Failure Mode Debugging

### Common Issues and Diagnosis

**Problem: r(k) < 0.5 (poor reconstruction)**
- Check: Did optimization converge? (plot loss)
- Check: Are gradients flowing? (gradient norm > 0?)
- Check: Is forward model correct? (test on known solution)
- Check: Is noise level too high? (try lower noise)

**Problem: Loss is NaN**
- Check: Numerical overflow in forward model
- Check: Learning rate too high
- Check: Division by zero in loss function
- Solution: Add gradient clipping, reduce learning rate

**Problem: Reconstruction has artifacts**
- Check: Boundary conditions (periodic?)
- Check: Grid resolution sufficient?
- Check: Observation model bugs
- Check: Prior too strong?

**Problem: Slow convergence**
- Check: Learning rate too small
- Check: Poor initialization
- Check: Need gradient preconditioning
- Solution: Tune optimizer hyperparameters

**Problem: Results not reproducible**
- Check: Random seeds set everywhere
- Check: Deterministic JAX operations
- Check: No racing conditions (if parallel)

---

## Continuous Validation

### Automated Testing

Run on every commit (CI):
```bash
pytest tests/ --cov=src/diffcosmo --cov-report=html
black --check src/ tests/
mypy src/
```

### Nightly Benchmarks

Run expensive tests overnight:
- Full 3D reconstruction benchmark
- Multi-GPU scaling tests
- Long MCMC chains for convergence
- Memory profiling

### Regression Testing

Maintain a suite of "gold standard" results:
- Known test cases with saved outputs
- Compare new code against baseline
- Flag any differences > tolerance

---

## Summary

**Validation is not optional.** Every feature must be validated before merging. When in doubt, add more tests.

**Key principles:**
1. **Test early, test often** — Catch bugs before they propagate
2. **Multiple metrics** — No single metric tells the whole story
3. **Visual inspection** — Trust your eyes, not just numbers
4. **Reproducibility** — If it's not reproducible, it's not science
5. **Document failures** — Failed tests teach us as much as passing ones

**For contributors:**
- Add tests for every new function
- Run validation suite before PRs
- Document validation results in notebooks
- Report any suspicious behavior

**For users:**
- Run validation on your own data
- Don't trust results without validation
- Report issues if validation fails

---

This is a living document. As we discover new failure modes or validation strategies, update this file.