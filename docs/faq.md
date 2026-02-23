# Beginner FAQ

A short FAQ for newcomers to this project.

## 1. Do I need a physics background to start?
No. Start with `docs/beginner_guide.md`, then run the toy script and inspect the outputs.

## 2. Is this a full production cosmology pipeline?
Not yet. The current implementation is a validated 2D differentiable MVP.

## 3. What does “field-level” mean here?
It means reconstructing the whole grid/field, not only summary metrics. See `docs/keywords.md`.

## 4. What does “gradient-based” mean in practice?
The code computes derivatives of the loss with respect to the initial field and uses them to improve reconstruction iteratively.

## 5. Why use JAX?
JAX gives autodiff, JIT acceleration, and a clean functional style useful for differentiable physics.

## 6. What is the main script to run?
`uv run --active python scripts/run_toy_2d.py --profile default`

## 7. How do I know if reconstruction is working?
Check output metrics and plots:
- loss should decrease
- cross-correlation should improve
- reconstructed field should visually align better with truth.

## 8. Where are settings controlled?
`data/toy/config.yaml`.

## 9. What are the next milestones?
- strengthen 2D validation further
- extend robustly to 3D
- add richer observation models
- add posterior sampling workflows.

## 10. I want the shortest onboarding path. What should I do?
1. Read `docs/beginner_guide.md`
2. Read `docs/keywords.md`
3. Run tests: `uv run --active pytest -q`
4. Run toy script
5. Open notebooks.
