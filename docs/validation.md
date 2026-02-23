# Validation Guide

This document defines what "working" means for this project.

The goal is simple: do not trust a reconstruction just because the code ran.
We validate both numerical behavior and scientific usefulness.

## What we validate

## 1) Numerical correctness

- no NaN/Inf in forward pass, gradients, or optimizer updates
- loss behaves sensibly during optimization
- deterministic behavior with fixed seeds

## 2) Reconstruction quality

- reconstructed initial field should be closer to truth than the initial guess
- key metrics should improve consistently across runs

## Core metrics

## Cross-correlation

Use `utils.cross_correlation(true, recon)`.

Interpretation:

- `1.0` means perfect agreement
- `0.0` means no linear correlation
- negative values mean anti-correlation

For toy reconstruction, we expect correlation to improve during optimization.

## Loss reduction

Use `loss_history` from `reconstruct_map`.

Healthy behavior:

- loss generally trends downward
- final loss is substantially lower than initial loss
- no sudden divergence to NaN/Inf

## Power spectrum comparison

Use `utils.compute_power_spectrum` on true and reconstructed fields.

We do not need exact match at every bin in toy runs, but broad scale behavior should align better after reconstruction.

## Practical acceptance checks (current 2D MVP)

A run is considered healthy if:

- tests pass: `uv run --active pytest -q`
- toy script completes without numerical errors
- final loss is lower than initial loss
- final cross-correlation improves vs initial guess

## Recommended validation workflow

1. Run tests.
2. Run default toy profile.
3. Record metrics from stdout.
4. Inspect summary figure (`toy2d_summary.png`).
5. Change one config parameter and re-run.

Do this before and after major code changes.

## Typical failure patterns and what to check

## Loss increases or oscillates

Possible causes:

- learning rate too high
- unstable gradient scale

Try:

- lower `opt.lr`
- enable stronger clipping / adjust optimizer settings

## Gradients become NaN or Inf

Possible causes:

- divide-by-zero in spectral operations
- unstable update step

Try:

- add/raise eps safeguards
- inspect force and loss components separately

## Correlation does not improve

Possible causes:

- observation noise too high for current setup
- weak optimization settings
- mismatch between forward and loss assumptions

Try:

- reduce noise for diagnostic runs
- increase optimization steps
- inspect mask handling and prior weight

## Reproducibility checklist

- run from same config profile
- keep seeds fixed
- log key metrics for each run
- compare figure outputs and scalar metrics together

## Automation

Current automated checks are in `tests/` and include:

- module-level unit tests
- end-to-end smoke test

Use this as your baseline gate before merging changes.
