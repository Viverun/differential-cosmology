# Differentiable Cosmology: The Inverse Universe

> 🚧 **Early Development** — 2D toy implementation in progress

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   [Placeholder: Workflow diagram or example comparison]     │
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

**→ New here?** See [**docs/getting_started.md**](docs/getting_started.md) for a complete hands-on tutorial.

---

## Prerequisites

**Ready to dive in?** Check [**PREREQUISITES.md**](PREREQUISITES.md) for a detailed guide on required knowledge and learning resources.

**TL;DR:** You should be comfortable with:
- Python and NumPy
- Basic calculus (derivatives, gradients)
- Fourier transforms (conceptually)
- Basic cosmology or willingness to learn

**New to some of these?** The prerequisites doc includes a learning roadmap and resources to get you started!

---

## Features

- ✅ 2D differentiable particle-mesh simulation (3D in progress)
- ✅ Gaussian initial conditions with power spectrum priors
- ✅ MAP inference via gradient-based optimization
- 🚧 Posterior sampling (flows/score models)
- 🚧 Realistic observation models

---

## Goals

Build differentiable cosmological forward models, demonstrate gradient-based MAP reconstruction, enable posterior inference with learned samplers, and provide reproducible experiments with clear validation.

---

## Documentation

**New to the project?** → Start with [**docs/getting_started.md**](docs/getting_started.md) for a hands-on tutorial

**Want to contribute?** → See [**CONTRIBUTING.md**](CONTRIBUTING.md) for guidelines

**Need more depth?**
- [**PREREQUISITES.md**](PREREQUISITES.md) — Required knowledge and learning resources
- [**docs/overview.md**](docs/overview.md) — Scientific and algorithmic foundations
- [**docs/architecture.md**](docs/architecture.md) — Software design and code structure
- [**docs/validation.md**](docs/validation.md) — Metrics and quality assurance

---

## Repository Structure

```
differentiable-cosmology/
├── src/diffcosmo/    # Core library
├── notebooks/        # Interactive tutorials
├── scripts/          # CLI tools
├── tests/            # Unit tests
└── docs/             # Documentation
```

See [**docs/architecture.md**](docs/architecture.md) for detailed code structure.

---

## Validation

We measure reconstruction quality using cross-correlation (target: r > 0.9), power spectrum recovery (<10% error), and stable optimization. See [**docs/validation.md**](docs/validation.md) for detailed metrics and testing strategy.

---

## Roadmap

**✅ Phase 0:** Setup — Repository scaffolding, 2D forward model  
**🚧 Phase 1:** 2D Toy (Current) — Differentiable PM, reconstruction, validation  
**📋 Phase 2:** 3D Realistic — Small volumes, observation models, MAP  
**📋 Phase 3:** Posterior Inference — MCMC, flows, score models  
**📋 Phase 4:** Scale & Realism — Larger volumes, baryonic physics, multi-GPU

---

## Scope

**In scope:** 2D/3D differentiable PM simulations, MAP reconstruction, posterior sampling methods, reproducible experiments.

**Out of scope (MVP):** Full hydrodynamics, production-scale surveys, petabyte simulations.

See [**docs/overview.md**](docs/overview.md) for complete scope and design decisions.

---

## Who This Is For

**Cosmologists** — Field-level inference tools  
**ML Researchers** — Physics-informed differentiable simulation testbed  
**Students** — Introduction to inverse problems in cosmology  
**Engineers** — Scalable scientific software patterns

---

## Contributing

Contributions are welcome! See [**CONTRIBUTING.md**](CONTRIBUTING.md) for guidelines on code style, testing, and pull requests.

**Good first areas:** Documentation, testing, utility functions, validation metrics.

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

Built on JAX-based cosmology tools (FlowPM, JaxPM) and field-level inference frameworks (BORG, ELUCID).