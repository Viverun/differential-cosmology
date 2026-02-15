# Prerequisites

This document helps you understand what knowledge and skills you'll need to work on this project. We've organized prerequisites by topic and marked them as **Essential**, **Important**, or **Nice to Have** so you can prioritize your learning.

Don't be intimidated if you don't know everything! This is a learning opportunity. Focus on the essentials first and build up from there.

---

## Quick Self-Assessment

**You're ready to start if you:**
- ✅ Can write and debug Python code comfortably
- ✅ Understand basic calculus (derivatives, gradients)
- ✅ Know what a Fourier transform does conceptually
- ✅ Have some intuition about probability and uncertainty
- ✅ Are curious about cosmology and willing to learn

**You'll have an easier time if you also:**
- Have used NumPy or similar array libraries
- Understand gradient descent and backpropagation
- Have run code on a GPU before
- Know basic Git workflows

---

## Mathematics & Physics

### Essential (Must Know)

**Calculus & Linear Algebra**
- What you need: Derivatives, gradients, chain rule, matrix operations
- Why: The entire project is about computing gradients through simulations
- Key concepts: ∂f/∂x, Jacobians, matrix-vector products
- Learn: Khan Academy Calculus, 3Blue1Brown Linear Algebra series

**Fourier Analysis (Conceptual)**
- What you need: What Fourier transforms do (convert spatial ↔ frequency domain)
- Why: PM simulations use FFTs for efficient force computation
- Key concepts: Frequency space, power spectra, convolution theorem
- Learn: "But what is the Fourier Transform?" (3Blue1Brown video)

**Basic Probability**
- What you need: Probability distributions, mean/variance, Gaussian distributions
- Why: Initial conditions are random fields; inference is about probability
- Key concepts: P(A|B), expectation, likelihood, priors
- Learn: Khan Academy Probability, first chapters of "Probabilistic Machine Learning"

### Important (Should Know)

**Cosmology Basics**
- What you need: What the universe is made of, how structure forms, what dark matter is
- Why: You're simulating the universe! Context helps debugging and validation
- Key concepts: Expansion, density fluctuations, gravity, large-scale structure
- Learn: "Cosmology for Everyone" (YouTube lectures), Ryden's "Introduction to Cosmology"

**Statistical Inference**
- What you need: Maximum likelihood, Bayesian inference, posterior distributions
- Why: The inverse problem is statistical inference
- Key concepts: MAP estimation, posterior = likelihood × prior, uncertainty quantification
- Learn: Chapters 1-3 of "Statistical Rethinking" or "Bayesian Data Analysis"

**Optimization Theory**
- What you need: Gradient descent, convexity, local minima, learning rates
- Why: MAP reconstruction uses gradient-based optimization
- Key concepts: Loss functions, convergence, momentum, Adam optimizer
- Learn: Stanford CS229 notes on optimization, "Deep Learning" book Ch. 4

### Nice to Have

**Advanced Cosmology**
- Power spectra, correlation functions, redshift-space distortions, halo models
- Helps with: Understanding validation metrics, designing better observation models
- Learn: Dodelson's "Modern Cosmology", Mo, van den Bosch & White

**Hamiltonian Dynamics**
- Phase space, symplectic integrators, conservation laws
- Helps with: Understanding PM evolution, checkpointing strategies
- Learn: Goldstein's "Classical Mechanics" Ch. 9-10

**Information Theory**
- KL divergence, entropy, mutual information
- Helps with: Understanding learned samplers, amortized inference
- Learn: Cover & Thomas "Elements of Information Theory" Ch. 1-2

---

## Programming & Software Engineering

### Essential (Must Know)

**Python (Intermediate Level)**
- What you need: Functions, classes, numpy arrays, loops, debugging
- Why: The entire codebase is Python
- Key skills: Read error messages, write small scripts, use print debugging
- Learn: "Python for Data Analysis" (McKinney), official Python tutorial

**NumPy Basics**
- What you need: Array creation, indexing, broadcasting, basic operations
- Why: All scientific computing in Python uses NumPy conventions
- Key skills: Reshape arrays, understand axis arguments, vectorize operations
- Learn: NumPy quickstart tutorial, "From Python to Numpy" (free online)

**Version Control (Git)**
- What you need: clone, commit, push, pull, branching basics
- Why: Collaboration and tracking changes
- Key skills: Make commits, sync with GitHub, resolve simple conflicts
- Learn: GitHub's Git tutorial, "Pro Git" book (free online)

### Important (Should Know)

**JAX Fundamentals**
- What you need: How JAX differs from NumPy, grad(), jit(), vmap()
- Why: JAX is our autodiff framework
- Key concepts: Functional programming, pure functions, automatic differentiation
- Learn: Official JAX quickstart, "JAX 101" tutorials

**Debugging & Testing**
- What you need: Using debuggers, writing unit tests, reading tracebacks
- Why: Scientific code has subtle bugs; testing catches them early
- Key skills: pytest basics, assert statements, test-driven development
- Learn: "Python Testing with pytest" (Okken), VS Code debugging guide

**Command Line Basics**
- What you need: Navigate directories, run scripts, environment variables
- Why: Running experiments, managing environments
- Key skills: cd, ls, python script.py, pip install
- Learn: "The Linux Command Line" (Shotts), terminal tutorials

### Nice to Have

**Advanced Python**
- Decorators, context managers, type hints, dataclasses
- Helps with: Reading and contributing to the codebase more effectively
- Learn: "Fluent Python" (Ramalho)

**Performance Profiling**
- Using profilers, understanding bottlenecks, memory usage
- Helps with: Optimization, debugging slow code
- Learn: Python profiling tutorials, line_profiler documentation

**GPU Programming Concepts**
- CUDA basics, parallel computing, memory hierarchy
- Helps with: Understanding performance, multi-GPU scaling
- Learn: NVIDIA CUDA tutorials, "CUDA by Example"

---

## Machine Learning & Inference

### Essential (Must Know)

**Gradient Descent & Backpropagation**
- What you need: How gradient descent works, what backpropagation does
- Why: The core of our inference method
- Key concepts: Learning rate, loss function, gradients flow backward
- Learn: 3Blue1Brown neural network series, Stanford CS231n lectures 1-4

**Automatic Differentiation**
- What you need: How autodiff differs from symbolic/numerical derivatives
- Why: Understanding JAX's grad() and when it works/fails
- Key concepts: Computation graphs, forward/reverse mode, chain rule
- Learn: "Automatic Differentiation in Machine Learning" (Baydin et al. 2018)

### Important (Should Know)

**Neural Networks Basics**
- What you need: What a neural network is, layers, activations, training loop
- Why: Helpful mental model for thinking about differentiable pipelines
- Key concepts: Forward pass, backward pass, parameters vs. hyperparameters
- Learn: Fast.ai course, "Neural Networks and Deep Learning" (Nielsen)

**Probabilistic Inference**
- What you need: Likelihood, prior, posterior, sampling vs. optimization
- Why: Understanding MAP vs. full posterior, why we need MCMC/flows
- Key concepts: Bayes' rule, marginalization, expectation under posterior
- Learn: "Probabilistic Machine Learning" (Murphy) Ch. 1-3

### Nice to Have

**Normalizing Flows**
- Invertible transformations, change of variables, flow-based generative models
- Helps with: Understanding amortized samplers (Phase 3 of project)
- Learn: "Normalizing Flows Tutorial" (Papamakarios et al. 2021)

**Score-Based Models**
- Diffusion models, score matching, Langevin dynamics
- Helps with: Alternative approach to posterior sampling
- Learn: "Generative Modeling by Estimating Gradients" (Yang Song blog)

**Markov Chain Monte Carlo**
- Metropolis-Hastings, Hamiltonian Monte Carlo, convergence diagnostics
- Helps with: Posterior sampling, understanding uncertainty
- Learn: "A Conceptual Introduction to HMC" (Betancourt 2017)

---

## Scientific Computing & Numerical Methods

### Essential (Must Know)

**FFT Basics (Usage)**
- What you need: How to call FFT functions, what they return
- Why: PM simulations are built on FFTs
- Key concepts: np.fft.fftn(), frequency ordering, Nyquist frequency
- Learn: NumPy FFT tutorial, "Understanding the FFT" (Jake VanderPlas blog)

**Numerical Integration Concepts**
- What you need: Euler method, timesteps, stability
- Why: PM evolution is time integration of ODEs
- Key concepts: Explicit vs. implicit, accuracy vs. stability
- Learn: "Numerical Methods for ODEs" tutorial, Wikipedia on Euler method

### Important (Should Know)

**Interpolation Methods**
- What you need: Grid-to-particle and particle-to-grid mapping
- Why: PM simulations assign densities and forces between grids and particles
- Key concepts: CIC (Cloud-In-Cell), nearest-grid-point, higher-order schemes
- Learn: Hockney & Eastwood "Computer Simulation Using Particles" Ch. 5

**Memory Management**
- What you need: In-place operations, memory allocation, avoiding copies
- Why: Large arrays (256³ cells) can exhaust memory quickly
- Key concepts: Views vs. copies, out parameter, memory profiling
- Learn: NumPy memory layout guide, JAX memory management docs

### Nice to Have

**Symplectic Integrators**
- Leap-frog, Verlet, higher-order schemes, reversibility
- Helps with: Understanding PM evolution choices, checkpointing strategies
- Learn: "Geometric Numerical Integration" (Hairer et al.)

**Parallel Computing**
- Vectorization, SIMD, GPU parallelism, distributed computing
- Helps with: Multi-GPU scaling, performance optimization
- Learn: "Parallel Programming in Python" tutorials

---

## Domain-Specific Knowledge

### Essential (Must Know)

**N-Body Simulations (Conceptual)**
- What you need: What cosmological simulations do, particles represent dark matter
- Why: PM is a type of N-body method
- Key concepts: Collisionless dynamics, structure formation, resolution limits
- Learn: "Cosmological N-body Simulations" reviews, YouTube lectures

### Important (Should Know)

**Power Spectra**
- What you need: What P(k) means, linear vs. nonlinear, shape interpretation
- Why: Our main validation metric
- Key concepts: Wavenumber k, variance σ², correlation function ↔ power spectrum
- Learn: Cosmology textbooks Ch. on structure formation, online lectures

**Observational Cosmology**
- What you need: How galaxy surveys work, what they measure, systematic errors
- Why: Designing realistic observation models
- Key concepts: Redshift, selection functions, photometric vs. spectroscopic
- Learn: "Observational Cosmology" lectures, survey documentation (SDSS, DES)

### Nice to Have

**Halo Models**
- Halo mass function, HOD, bias, halo occupation distribution
- Helps with: Phase 2 observation models
- Learn: Cooray & Sheth "Halo Models" (2002), Mo & White review

**Redshift-Space Distortions**
- Kaiser effect, Finger-of-God, velocity fields
- Helps with: Realistic observation operators
- Learn: Hamilton "Redshift-space distortions" review

---

## Practical Skills Checklist

Before diving into the code, make sure you can:

**Week 1 Readiness:**
- [ ] Run Python scripts from command line
- [ ] Install packages with pip
- [ ] Create and manipulate NumPy arrays
- [ ] Compute gradients of simple functions using JAX
- [ ] Clone a GitHub repo and run existing notebooks
- [ ] Understand what a loss function is

**Month 1 Readiness:**
- [ ] Write basic unit tests with pytest
- [ ] Debug Python code using print statements or a debugger
- [ ] Visualize 2D arrays as images using matplotlib
- [ ] Perform FFTs on multidimensional arrays
- [ ] Understand gradient descent optimization conceptually
- [ ] Explain what initial conditions and structure formation mean

**Full Project Readiness:**
- [ ] Implement a simple optimization loop with JAX
- [ ] Understand forward vs. inverse problems
- [ ] Know when to use MAP vs. full posterior sampling
- [ ] Can read and interpret power spectra
- [ ] Profile code to find bottlenecks
- [ ] Comfortable with Git branching and pull requests

---

## Learning Roadmap

**If you're starting from scratch:**

**Phase 1 (1-2 months): Foundations**
1. Solidify Python + NumPy skills
2. Learn JAX basics (grad, jit, vmap)
3. Study gradient descent and backpropagation
4. Understand Fourier transforms conceptually
5. Learn basic cosmology (what the universe is made of)

**Phase 2 (1-2 months): Core Concepts**
1. Deep dive into automatic differentiation
2. Study Bayesian inference and MAP estimation
3. Learn about N-body simulations and PM methods
4. Practice using FFTs for physics problems
5. Understand power spectra and correlation functions

**Phase 3 (ongoing): Project-Specific**
1. Work through the toy 2D notebooks
2. Read the technical overview (docs/overview.md)
3. Study the codebase module by module
4. Implement small features or fix bugs
5. Read papers on differentiable cosmology

---

## Recommended Resources

### Online Courses (Free)
- **Python:** Kaggle Python course, Real Python tutorials
- **Scientific Computing:** SciPy lecture notes
- **Machine Learning:** Fast.ai, Stanford CS229
- **Cosmology:** Cosmology lectures on YouTube (Ned Wright, Sean Carroll)

### Books (Accessible)
- **Python:** "Python Crash Course" (Matthes)
- **NumPy:** "Elegant SciPy" (free online)
- **ML Basics:** "Deep Learning" (Goodfellow et al., free online)
- **Cosmology:** Ryden "Introduction to Cosmology"
- **Inference:** "Statistical Rethinking" (McElreath)

### Papers (Start Here)
- Modi et al. (2021) - "FlowPM: Distributed TensorFlow Implementation of Particle-Mesh Cosmology"
- Li et al. (2021) - "Differentiable Cosmological Simulation"
- Villaescusa-Navarro et al. (2022) - "The Quijote Simulations"

### Documentation (Essential)
- JAX documentation and tutorials
- NumPy user guide
- pytest documentation
- Git tutorials

---

## Getting Help

**Don't know where to start?**
- Open a GitHub Discussion describing your background
- We'll suggest a personalized learning path

**Stuck on a specific concept?**
- Check the docs/ folder for explanations
- Ask in GitHub Issues with the "question" label
- Pair program with other contributors

**Want to contribute but feeling underprepared?**
- Start with documentation improvements (always needed!)
- Review notebooks and suggest clarifications
- Test installation on different platforms
- Work on visualization and plotting code

---

## Final Thoughts

This project sits at the intersection of cosmology, machine learning, and scientific computing. **Nobody knows everything on this list.** The goal is to:

1. Identify what you already know (celebrate that!)
2. Recognize gaps that are blocking you (focus here)
3. Make a learning plan (start small)
4. Contribute while learning (best way to learn!)

We're building this together. Questions are welcome, and explaining concepts to others is one of the best ways to learn. Welcome to the project!