# Simple explanation — *Differentiable Cosmology: The “Inverse” Universe* (plain language)

Nice — here’s the whole plan translated into simple, non-technical language so you can explain it to a friend (or use it as a quick pitch).

## What we’re trying to do (one-sentence)

We want to build a system that can look at the universe as it is now (galaxies, maps, observations) and work backwards to figure out what the tiny ripples in the very early universe looked like — using math that lets us take derivatives through the whole simulation so we can “learn” the initial state.

## Why that’s cool and useful

* Instead of guessing initial conditions and forward-simulating millions of times, we use gradients to search directly for the initial pattern that best explains today.
* This gives stronger, more detailed scientific answers about how the universe evolved, and better estimates of uncertainties (how sure we are).
* It’s like training a giant neural network where the “weights” are the early-universe ripples.

## What the project will build (simple parts)

1. **A differentiable simulator** — a physics engine you can compute derivatives through. This is the core: run the universe forward, but in a way that supports backtracking. We’ll use autodiff tools such as JAX for that.
2. **An observation model** — fake how a telescope would see the simulated universe (add noise, masks, selection effects) so our comparisons are realistic.
3. **An inference engine** — optimization and sampling tools that use gradients to find the best initial conditions and, ideally, samples from the probable set of initial conditions.
4. **Validation & visuals** — tests and plots showing how well our reconstructed early universe matches the true one in simulations.

## How we’ll proceed (phases, in plain terms)

* **Phase 1 (toy):** Start very small — 2D world, tiny grid. Make sure we can take gradients and recover the known initial pattern. This is a “proof of life.”
* **Phase 2 (realistic-ish 3D):** Move to a small 3D volume, add a simple model for galaxies and how telescopes see them, then try to recover initial conditions for that mock survey.
* **Phase 3 (posterior ideas):** Go beyond a single best guess: build methods that give many plausible early-universe maps (uncertainty quantification), using learned samplers or gradient-based sampling.
* **Phase 4 (scale & realism):** Add more real-world complications (better galaxy models, observational quirks, baryonic physics) and scale up to bigger boxes / more compute.

## What success looks like (plain checks)

* We can take a simulated universe, hide the initial state, and reconstruct it well enough that the reconstructed and true initial fields look and behave similarly (measured with simple stats like correlation and power spectra).
* Our method runs end-to-end on a single GPU for the toy version and is documented so others can reproduce it.
* We can show uncertainty estimates that are sensible (not overconfident).

## Big challenges (in normal words)

* **Memory and compute:** Running a time-looped simulation and backpropagating through it uses a lot of memory. We’ll need tricks to recompute states or use reversible steps.
* **Observational mess:** Real telescope data have gaps, errors, and selection quirks — if we ignore those, the reconstruction will be biased.
* **Scale:** Doing full posterior sampling for a realistic volume is extremely expensive; we’ll focus on best-fit solutions and learned samplers instead of naive brute-force sampling.

## What you (as a computer engineer & ML person) will actually do

* Implement the differentiable simulator parts and make them fast and memory-efficient.
* Design and train ML samplers / flows or use gradient-based optimizers for reconstruction.
* Build pipelines and tooling so everything runs reproducibly and can later be scaled to many GPUs.

## First three practical steps you can do today

1. Make a small GitHub repo with the project structure (notebooks, src, data).
2. Build the 2D toy notebook: create a random initial pattern, run a tiny simulator forward, then try to recover the initial pattern by gradient descent.
3. Plot target vs reconstructed and write a short README describing what worked and what failed.

---