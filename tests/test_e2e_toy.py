"""End-to-end smoke test for the toy 2D pipeline."""

import math

from scripts.run_toy_2d import run_experiment


def test_toy_2d_pipeline_smoke(tmp_path):
    config = {
        "grid_shape": [16, 16],
        "sim": {
            "n_steps": 4,
            "dt": 0.1,
            "checkpointing": "none",
            "cosmology": {"Omega_m": 0.3, "init_disp_scale": 0.2},
        },
        "prior": {
            "amplitude": 1.0,
            "index": -2.0,
            "lambda_prior": 1.0e-2,
            "eps": 1.0e-6,
        },
        "obs": {
            "noise_std": 0.1,
            "use_mask": True,
            "mask_frac": 0.8,
            "seed": 2,
        },
        "opt": {
            "lr": 0.05,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "max_iters": 30,
            "early_stop_patience": 10,
            "early_stop_min_delta": 1.0e-6,
        },
        "seeds": {"truth": 0, "init": 1},
    }

    metrics = run_experiment(config, tmp_path)

    assert math.isfinite(metrics["initial_loss"])
    assert math.isfinite(metrics["final_loss"])
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert (tmp_path / "toy2d_summary.png").exists()
