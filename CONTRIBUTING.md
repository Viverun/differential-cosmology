# Contributing to Differentiable Cosmology

Thank you for your interest in contributing! This project aims to make field-level cosmological inference practical and accessible. Contributions of all kinds are welcome—code, documentation, bug reports, examples, and ideas.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)
- [Getting Help](#getting-help)

---

## Getting Started

### Prerequisites

Before contributing, make sure you're familiar with:
- Python and NumPy basics
- JAX fundamentals (grad, jit, vmap)
- Basic cosmology concepts
- Git and GitHub workflows

See [PREREQUISITES.md](PREREQUISITES.md) for detailed requirements and learning resources.

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
```bash
git clone https://github.com/YOUR-USERNAME/differentiable-cosmology.git
cd differentiable-cosmology
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e .
pip install -r requirements-dev.txt
```

4. **Verify installation:**
```bash
pytest tests/
python -c "import diffcosmo; print('Success!')"
```

5. **Set up pre-commit hooks (optional but recommended):**
```bash
pip install pre-commit
pre-commit install
```

---

## Ways to Contribute

### Code Contributions

**Good first issues:**
- Add utility functions (plotting, I/O helpers)
- Improve error messages
- Write additional unit tests
- Optimize existing functions
- Add type hints

**Intermediate contributions:**
- Implement new observation operators
- Add new optimization algorithms
- Improve checkpointing strategies
- Extend validation metrics

**Advanced contributions:**
- 3D PM evolution implementation
- Posterior sampling methods (HMC, flows)
- Multi-GPU support
- Differentiable hydrodynamics

### Non-Code Contributions

**Documentation:**
- Fix typos or unclear explanations
- Add examples to docstrings
- Create tutorial notebooks
- Improve installation instructions
- Translate documentation

**Science:**
- Validate results against literature
- Suggest better priors or observation models
- Propose new validation metrics
- Compare with other methods

**Community:**
- Answer questions in Discussions
- Review pull requests
- Report bugs with minimal reproducible examples
- Share use cases and results

---

## Development Workflow

### 1. Create a Branch

Always work on a feature branch, not `main`:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

**Branch naming conventions:**
- `feature/` — new features (e.g., `feature/rsd-operator`)
- `fix/` — bug fixes (e.g., `fix/gradient-nan`)
- `docs/` — documentation only (e.g., `docs/update-readme`)
- `test/` — test additions (e.g., `test/pm-convergence`)
- `refactor/` — code refactoring (e.g., `refactor/loss-module`)

### 2. Make Your Changes

**Keep changes focused:**
- One logical change per PR
- Avoid mixing refactoring with feature additions
- If you notice unrelated issues, create separate PRs

**Write good commit messages:**
```bash
# Good
git commit -m "Add redshift-space distortion operator to observe.py"
git commit -m "Fix gradient NaN issue in loss computation"
git commit -m "Improve power spectrum plotting with error bars"

# Bad
git commit -m "fix stuff"
git commit -m "wip"
git commit -m "more changes"
```

**Commit message format:**
```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how (code shows how).

- Bullet points are fine
- Reference issues: Fixes #123
- Reference PRs: Related to #456
```

### 3. Keep Your Branch Updated

Regularly sync with `main` to avoid conflicts:

```bash
git checkout main
git pull upstream main
git checkout your-branch
git rebase main
```

If you encounter conflicts, resolve them and continue:
```bash
# Fix conflicts in your editor
git add resolved-file.py
git rebase --continue
```

### 4. Test Your Changes

Before pushing, ensure:
```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/ tests/

# Check type hints (if using mypy)
mypy src/

# Run a quick integration test (if applicable)
python scripts/run_toy_2d.py
```

### 5. Push and Create a Pull Request

```bash
git push origin your-branch
```

Then open a pull request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference to related issues (`Fixes #123`)
- Screenshots/plots if relevant
- Checklist of completed items (see PR template)

---

## Code Standards

### Style Guide

We follow **PEP 8** with some modifications:

**Line length:** 100 characters (not 79)
**String quotes:** Double quotes `"` preferred
**Imports:** Organized with `isort`
**Formatting:** Automated with `black`

### Code Formatting

Format code with Black before committing:

```bash
black src/ tests/ scripts/
```

Configuration is in `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ['py39']
```

### Naming Conventions

**Functions and variables:** `snake_case`
```python
def compute_power_spectrum(field):
    n_grid = field.shape[0]
    ...
```

**Classes:** `PascalCase` (rare in this codebase)
```python
class FlowSampler:
    ...
```

**Constants:** `UPPER_SNAKE_CASE`
```python
DEFAULT_LEARNING_RATE = 1e-3
MAX_ITERATIONS = 10000
```

**Private functions:** Leading underscore `_function_name`
```python
def _internal_helper(x):
    # Only used within this module
    ...
```

### Type Hints

Use type hints for function signatures:

```python
import jax.numpy as jnp
from typing import Tuple, Optional, Callable

def evolve_pm(
    initial_field: jnp.ndarray,
    n_steps: int,
    cosmology: dict,
    checkpointing: str = 'uniform'
) -> jnp.ndarray:
    """Evolve field forward in time."""
    ...
```

For complex types, define aliases:
```python
from typing import Dict, Any

CosmologyParams = Dict[str, float]
LossFn = Callable[[jnp.ndarray], Tuple[float, Dict[str, Any]]]
```

### Docstrings

Use **NumPy-style** docstrings:

```python
def compute_cross_correlation(field1, field2):
    """Compute cross-correlation coefficient between two fields.
    
    The cross-correlation measures how well field2 matches field1 at each
    wavenumber k in Fourier space.
    
    Parameters
    ----------
    field1 : jnp.ndarray
        First field (ground truth), shape (N, N) or (N, N, N)
    field2 : jnp.ndarray
        Second field (reconstruction), same shape as field1
    
    Returns
    -------
    k_bins : jnp.ndarray
        Wavenumber bins, shape (n_bins,)
    r_k : jnp.ndarray
        Cross-correlation coefficient at each k, shape (n_bins,)
    
    Examples
    --------
    >>> true_field = generate_gaussian_field((128, 128), seed=42)
    >>> recon_field = reconstruct(observations)
    >>> k, r = compute_cross_correlation(true_field, recon_field)
    >>> print(f"Correlation at large scales: r(k_min) = {r[0]:.3f}")
    
    Notes
    -----
    The cross-correlation is defined as:
    
    .. math::
        r(k) = \\frac{\\langle \\delta_1(k) \\delta_2^*(k) \\rangle}{
               \\sqrt{\\langle |\\delta_1(k)|^2 \\rangle \\langle |\\delta_2(k)|^2 \\rangle}}
    
    See Also
    --------
    compute_power_spectrum : Compute power spectrum of a single field
    """
    ...
```

**Minimum docstring requirements:**
- One-line summary
- Parameters with types and descriptions
- Returns with types and descriptions
- Examples for non-trivial functions

### JAX Best Practices

**Pure functions only:**
```python
# Good
def add_noise(field, noise_level, seed):
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, field.shape) * noise_level
    return field + noise

# Bad (side effects)
global_state = {}
def add_noise_bad(field, noise_level):
    global global_state  # Don't do this!
    noise = np.random.randn(*field.shape) * noise_level
    global_state['last_noise'] = noise
    return field + noise
```

**Use `jnp` not `np` inside functions:**
```python
import jax.numpy as jnp
import numpy as np  # Only for non-differentiable I/O

def my_function(x):
    return jnp.sin(x)  # Good, differentiable

# Use regular numpy only for I/O, plotting
data = np.load('data.npy')  # OK
```

**Avoid Python loops, use JAX control flow:**
```python
# Prefer vmap/scan over loops
def compute_many(fields):
    return jax.vmap(compute_single)(fields)  # Good

# For sequential operations, use lax.scan
def evolve_steps(state, steps):
    return jax.lax.scan(step_fn, state, steps)  # Good
```

---

## Testing Requirements

### Test Coverage

All new code must include tests. Aim for:
- **90%+ coverage** for core modules (`fields.py`, `pm.py`, `loss.py`)
- **80%+ coverage** for inference and utilities
- **100% coverage** for critical paths (gradient computation, loss functions)

Check coverage:
```bash
pytest --cov=src/diffcosmo --cov-report=html
open htmlcov/index.html  # View detailed report
```

### Types of Tests

**Unit tests** — Test individual functions:
```python
# tests/test_fields.py
def test_generate_gaussian_field_shape():
    """Field should have correct shape."""
    field = generate_gaussian_field((64, 64), seed=42)
    assert field.shape == (64, 64)

def test_generate_gaussian_field_deterministic():
    """Same seed should give same field."""
    field1 = generate_gaussian_field((64, 64), seed=42)
    field2 = generate_gaussian_field((64, 64), seed=42)
    assert jnp.allclose(field1, field2)
```

**Integration tests** — Test pipelines:
```python
# tests/test_integration.py
def test_forward_then_inverse():
    """Can we recover a known field?"""
    true_field = generate_gaussian_field((32, 32), seed=42)
    observations = forward_model(true_field)
    recon_field = inverse_model(observations)
    
    r = compute_cross_correlation(true_field, recon_field)
    assert r > 0.7  # Should recover most signal
```

**Gradient tests** — Verify autodiff:
```python
def test_loss_gradients():
    """Gradients should match finite differences."""
    params = jnp.ones((32, 32))
    
    # Analytic gradient
    grad_fn = jax.grad(loss_fn)
    analytic_grad = grad_fn(params)
    
    # Numerical gradient
    eps = 1e-5
    numerical_grad = jnp.zeros_like(params)
    for i in range(params.size):
        params_plus = params.at.flat[i].add(eps)
        params_minus = params.at.flat[i].add(-eps)
        numerical_grad[i] = (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * eps)
    
    assert jnp.allclose(analytic_grad, numerical_grad, rtol=1e-3)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_fields.py

# Run specific test
pytest tests/test_fields.py::test_generate_gaussian_field_shape

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/diffcosmo

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Test Fixtures

Use pytest fixtures for shared setup:

```python
# tests/conftest.py
import pytest
import jax.numpy as jnp

@pytest.fixture
def simple_field():
    """Small test field for unit tests."""
    return generate_gaussian_field((32, 32), seed=42)

@pytest.fixture
def cosmology_params():
    """Standard cosmology parameters."""
    return {
        'Omega_m': 0.3,
        'sigma_8': 0.8,
        'h': 0.7
    }

# Use in tests:
def test_something(simple_field, cosmology_params):
    result = my_function(simple_field, cosmology_params)
    assert result.shape == simple_field.shape
```

---

## Documentation Guidelines

### Code Documentation

**Every public function needs a docstring:**
- One-line summary
- Extended description if complex
- All parameters documented
- Return values documented
- Examples for non-trivial usage

**Module-level docstrings:**
```python
"""Observation operators for mock galaxy surveys.

This module provides functions to convert evolved density fields into
realistic galaxy observations, including Poisson sampling, survey masks,
and observational noise.

Key functions:
    sample_galaxies_poisson: Sample galaxy positions from density field
    apply_survey_mask: Apply geometric survey boundaries
    add_observation_noise: Add measurement uncertainties
"""
```

### Tutorial Notebooks

When adding notebooks:
- Clear title and introduction
- Explain what you'll demonstrate
- Show inputs and outputs
- Include visualizations
- Add narrative text between code cells
- End with summary and next steps

**Notebook structure:**
```markdown
# Tutorial: [Topic]

**Goal:** Learn how to [objective]

**Prerequisites:** [what you should know]

## Setup
[imports and configuration]

## Step 1: [First concept]
[explanation]
[code]
[visualization]

## Step 2: [Next concept]
...

## Summary
[what we learned]
[next steps]
```

### Documentation Files

When updating docs in `docs/`:
- Keep consistent style with existing docs
- Add concrete examples
- Update table of contents if adding sections
- Cross-reference related documents
- Use code blocks for examples

---

## Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guide (run `black`)
- [ ] All tests pass locally (`pytest`)
- [ ] New code has tests (check coverage)
- [ ] Docstrings added for new functions
- [ ] Type hints added
- [ ] No merge conflicts with `main`
- [ ] Notebook outputs cleared (if applicable)
- [ ] CHANGELOG.md updated (if significant change)

### PR Template

When opening a PR, include:

```markdown
## Description
Brief description of what this PR does.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Added X
- Modified Y
- Fixed Z

## Related Issues
Fixes #123
Related to #456

## Testing
How was this tested?
- Added unit tests for X
- Ran integration test Y
- Verified output manually

## Checklist
- [ ] Tests pass
- [ ] Code formatted with black
- [ ] Documentation updated
- [ ] Type hints added

## Screenshots (if applicable)
[Add before/after plots, UI changes, etc.]
```

### Review Process

1. **Automated checks** run (CI tests, linting)
2. **Maintainer review** (1-2 business days)
3. **Address feedback** with new commits
4. **Approval** and merge when ready

**During review, be prepared to:**
- Answer questions about design choices
- Make requested changes
- Add additional tests
- Improve documentation

**Be patient and respectful:**
- Reviews may take time
- Reviewers may request changes
- Discussion is part of the process
- We're all learning together

### After Merging

- Delete your feature branch
- Close related issues
- Update any dependent PRs
- Celebrate! 🎉

---

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment.

**Expected behavior:**
- Be respectful and considerate
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards other community members

**Unacceptable behavior:**
- Harassment, discrimination, or personal attacks
- Trolling, insulting comments, or sustained disruption
- Publishing others' private information
- Any conduct inappropriate in a professional setting

**Enforcement:**
Violations can be reported to [maintainer email]. All reports will be reviewed and investigated, and will result in a response deemed necessary and appropriate.

### Best Practices for Collaboration

**Asking questions:**
- Search existing issues/discussions first
- Provide context and what you've tried
- Include minimal reproducible examples
- Be specific about expected vs. actual behavior

**Reporting bugs:**
- Use the bug report template
- Include environment details (Python version, JAX version, OS)
- Provide code to reproduce the bug
- Include error messages and stack traces

**Suggesting features:**
- Open a Discussion first (not an issue)
- Explain the use case and motivation
- Consider implementation complexity
- Be open to feedback and alternatives

**Reviewing code:**
- Be constructive and kind
- Suggest improvements, don't demand them
- Explain the "why" behind requests
- Approve when ready, even if not perfect

---

## Getting Help

### Resources

**Documentation:**
- [PREREQUISITES.md](PREREQUISITES.md) — Required knowledge
- [docs/overview.md](docs/overview.md) — Scientific background
- [docs/architecture.md](docs/architecture.md) — Code structure
- [docs/validation.md](docs/validation.md) — Testing strategy

**Communication:**
- **GitHub Discussions** — Questions, ideas, general discussion
- **GitHub Issues** — Bug reports, feature requests
- **Email** — [maintainer email] for private concerns

### Common Questions

**Q: I'm new to JAX. Where should I start?**
A: Check PREREQUISITES.md for JAX resources. Start with the official JAX quickstart, then try modifying our 2D toy example.

**Q: I want to help but don't know cosmology well.**
A: Great! Focus on software contributions: testing, documentation, optimization, tooling. You'll learn cosmology along the way.

**Q: My tests are failing. What do I do?**
A: First, run tests locally to see detailed error messages. Check if your environment is correct. Ask in Discussions if stuck.

**Q: Can I contribute if I only have CPU (no GPU)?**
A: Absolutely! Most development work runs fine on CPU. The 2D toy example works on laptop CPU.

**Q: How long until my PR is reviewed?**
A: Usually 1-3 business days for initial review. Complex PRs may take longer. If it's been a week with no response, ping the PR.

**Q: I have an idea but not sure how to implement it.**
A: Open a Discussion! We'll brainstorm together and guide implementation.

---

## Recognition

Contributors will be:
- Listed in AUTHORS.md
- Acknowledged in release notes
- Credited in papers using this code (when appropriate)
- Given co-authorship for substantial contributions

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

Every contribution makes this project better. Whether you fix a typo, add a test, implement a feature, or just ask a good question—thank you for being part of this project.

**Happy coding!** 🚀