# Differentiable Cosmology: The Inverse Universe

> 🚧 **Early Development** — 2D toy implementation in progress

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   [Placeholder: Workflow diagram or example comparison]    │
│                                                             │
│   Initial Field → Forward Evolution → Observations          │
│                         ↓                                   │
│              Backprop & Reconstruct ←                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## What This Does

**Differentiable cosmology** treats the universe's evolution as a differentiable function, enabling gradient-based inference of initial conditions from observations.

**Traditional approach:**  
Guess initial conditions → simulate forward → compare to data → repeat millions of times

**This project:**  
Observe the universe → backpropagate through simulation → infer initial conditions directly

Think of it as training a neural network where the "weights" are the primordial density fluctuations of the early universe.

---

## Why This Matters

- 🎯 **Field-level inference** — reconstruct full density fields, not just summary statistics
- 📈 **Gradient-based optimization** — orders of magnitude faster than forward sampling
- 🧠 **Modern ML meets physics** — brings autodiff and learned samplers to cosmology
- 🔬 **Better science** — richer constraints on cosmological parameters and uncertainties

---

## Quick Start

### Installation

```bash
# Requirements: Python 3.9+, JAX
git clone https://github.com/yourusername/differentiable-cosmology.git
cd differentiable-cosmology
pip install -e .
```

### Run the 2D Toy Example

```bash
python scripts/run_toy_2d.py
```

**Expected output:** Plots showing forward evolution and gradient-based reconstruction of initial density field.

### Interactive Notebooks

Explore step-by-step:
- `notebooks/00_toy_2d_forward.ipynb` — Generate and evolve initial conditions
- `notebooks/01_toy_2d_inverse.ipynb` — Reconstruct via gradient descent

---

## Key Features

- ✅ Differentiable particle-mesh (PM) simulation (2D complete, 3D in progress)
- ✅ Gaussian random field initial conditions with power spectrum priors
- ✅ Simple observation models (noise, masking, selection effects)
- ✅ MAP inference via gradient-based optimization
- 🚧 Posterior sampling with learned models (flows/score-based)
- 🚧 3D realistic volumes and survey realism

---

## Project Goals

1. Build a **differentiable cosmological forward model** (toy → realistic scales)
2. Demonstrate **gradient-based MAP reconstruction** of initial density fields
3. Enable **posterior inference** using gradient-informed and amortized samplers
4. Provide **reproducible experiments** with clear validation metrics
5. Produce **extensible open-source tools** for the community

---

## Repository Structure

```
differentiable-cosmology/
├── notebooks/        # Interactive demos
├── src/diffcosmo/    # Core library (fields, PM solver, inference)
├── scripts/          # CLI entry points
├── data/             # Synthetic benchmark data
├── docs/             # Detailed methodology and validation
└── tests/            # Unit tests
```

See [`docs/architecture.md`](docs/architecture.md) for detailed structure.

---

## Validation & Success Criteria

We measure success through:
- **Cross-correlation** between true and reconstructed initial fields (target: r > 0.9 on large scales)
- **Power spectrum recovery** with <10% error on targeted wavenumbers
- **Stable optimization** with interpretable loss curves
- **Reproducibility** across different random seeds

Full validation details: [`docs/validation.md`](docs/validation.md)

---

## Current Status & Roadmap

**✅ Phase 0: Setup**  
Repository scaffolding, initial field generator, 2D forward model

**🚧 Phase 1: 2D Toy (Current)**  
Differentiable 2D PM, gradient-based reconstruction, validation

**📋 Phase 2: 3D Realistic (Next)**  
Small 3D volumes, observation models, MAP reconstruction

**📋 Phase 3: Posterior Inference**  
Gradient-informed MCMC, normalizing flows or score models

**📋 Phase 4: Scale & Realism**  
Larger volumes, baryonic physics, multi-GPU support

---

## Project Scope

### In Scope
- 2D and small 3D differentiable PM simulations
- Gradient-based MAP reconstruction
- At least one amortized posterior method
- Reproducible experiments and validation

### Out of Scope (for MVP)
- Full hydrodynamics / baryonic feedback
- Production-scale survey pipelines
- Petabyte-scale simulations

---

## Use Cases & Audience

**For cosmologists:** Tools for field-level inference and initial condition reconstruction

**For ML researchers:** Physics-informed differentiable simulation as a testbed for learned inference

**For students:** Hands-on introduction to inverse problems in cosmology

**For engineers:** Scalable scientific software patterns (autodiff, checkpointing, multi-GPU)

---

## Documentation

- [**Project Overview**](docs/overview.md) — Detailed methodology and architecture
- [**Validation Guide**](docs/validation.md) — Metrics and success criteria
- [**Contributing**](CONTRIBUTING.md) — How to contribute code or ideas

---

## Future Extensions

- Larger 3D volumes at higher resolution
- Differentiable hydrodynamics for baryonic realism
- Application to real survey data (SDSS, DES, Euclid)
- Joint inference of cosmological parameters and initial conditions

---

## Contributing

We welcome contributions! Areas where help is especially valuable:
- Performance optimization (memory, speed)
- Observation model realism
- Posterior sampling methods
- Documentation and tutorials

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{differentiable_cosmology,
  title = {Differentiable Cosmology: The Inverse Universe},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/differentiable-cosmology}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/differentiable-cosmology/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/differentiable-cosmology/discussions)
- **Email:** your.email@example.com

---

## Acknowledgments

This project builds on foundational work in differentiable simulation and field-level cosmology. Key inspirations include JAX-based cosmology tools (FlowPM, JaxPM) and field-level inference frameworks (BORG, ELUCID).