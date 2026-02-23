"""Run a complete 2D toy differentiable cosmology reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from diffcosmo.fields import generate_gaussian_field, power_law_power_spectrum
from diffcosmo.inference import reconstruct_map
from diffcosmo.loss import total_loss
from diffcosmo.observe import apply_observation_model, make_rectangular_mask
from diffcosmo.pm import evolve_pm
from diffcosmo.utils import compute_power_spectrum, cross_correlation


def _load_profile(config_path: Path, profile: str | None) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    profiles = config["profiles"]
    selected = profile or config.get("active_profile", "default")
    if selected not in profiles:
        raise ValueError(f"Unknown profile '{selected}'. Available: {sorted(profiles.keys())}")
    return profiles[selected]


def run_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_shape = tuple(config["grid_shape"])
    sim_config = config["sim"]
    prior_config = config["prior"]
    obs_params = config["obs"]
    opt_config = config["opt"]
    seeds = config["seeds"]

    power_fn = lambda k: power_law_power_spectrum(
        k,
        amplitude=float(prior_config["amplitude"]),
        index=float(prior_config["index"]),
    )

    theta_true = generate_gaussian_field(grid_shape, power_fn, seed=int(seeds["truth"]))
    delta_true = evolve_pm(
        theta_true,
        n_steps=int(sim_config["n_steps"]),
        dt=float(sim_config["dt"]),
        cosmology=sim_config["cosmology"],
        checkpointing=sim_config.get("checkpointing", "none"),
    )

    mask = None
    if bool(obs_params.get("use_mask", False)):
        mask = make_rectangular_mask(grid_shape, frac=float(obs_params.get("mask_frac", 0.8)))

    y_obs = apply_observation_model(
        delta_true,
        noise_std=float(obs_params["noise_std"]),
        mask=mask,
        seed=int(obs_params.get("seed", 0)),
    )

    theta_init = generate_gaussian_field(grid_shape, power_fn, seed=int(seeds["init"]))
    obs_config = {"noise_std": float(obs_params["noise_std"]), "mask": mask}

    initial_loss = float(total_loss(theta_init, y_obs, sim_config, prior_config, obs_config))
    result = reconstruct_map(
        y_obs=y_obs,
        theta_init=theta_init,
        sim_config=sim_config,
        prior_config=prior_config,
        obs_config=obs_config,
        opt_config=opt_config,
    )
    theta_map = result["theta_map"]
    loss_history = result["loss_history"]

    final_loss = float(loss_history[-1]) if loss_history.size else float("nan")
    initial_corr = float(cross_correlation(theta_true, theta_init))
    final_corr = float(cross_correlation(theta_true, theta_map))
    corr_improvement = final_corr - initial_corr
    loss_reduction_fraction = (initial_loss - final_loss) / max(initial_loss, 1e-8)

    if not jnp.isfinite(theta_map).all():
        raise RuntimeError("Reconstruction produced non-finite values.")

    _save_plots(
        output_dir=output_dir,
        theta_true=theta_true,
        theta_init=theta_init,
        theta_map=theta_map,
        loss_history=loss_history,
    )

    jnp.save(output_dir / "theta_true.npy", theta_true)
    jnp.save(output_dir / "theta_init.npy", theta_init)
    jnp.save(output_dir / "theta_map.npy", theta_map)
    jnp.save(output_dir / "loss_history.npy", loss_history)

    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_fraction": float(loss_reduction_fraction),
        "initial_corr": initial_corr,
        "final_corr": final_corr,
        "corr_improvement": float(corr_improvement),
        "n_iters": int(result["n_iters"]),
    }
    return metrics


def _save_plots(
    output_dir: Path,
    theta_true: jnp.ndarray,
    theta_init: jnp.ndarray,
    theta_map: jnp.ndarray,
    loss_history: jnp.ndarray,
) -> None:
    residual = theta_map - theta_true

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    vlim = float(jnp.max(jnp.abs(theta_true)))

    im0 = axes[0, 0].imshow(theta_true, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[0, 0].set_title("True initial field")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(theta_init, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[0, 1].set_title("Initial guess")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(theta_map, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[0, 2].set_title("Reconstructed field")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(residual, cmap="coolwarm")
    axes[1, 0].set_title("Residual (recon - true)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    if loss_history.size > 0:
        axes[1, 1].plot(loss_history)
        axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Loss history")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss")

    k_true, p_true = compute_power_spectrum(theta_true)
    k_map, p_map = compute_power_spectrum(theta_map)
    axes[1, 2].loglog(k_true, p_true + 1e-12, label="True")
    axes[1, 2].loglog(k_map, p_map + 1e-12, label="Reconstructed")
    axes[1, 2].set_title("Power spectrum")
    axes[1, 2].set_xlabel("k")
    axes[1, 2].set_ylabel("P(k)")
    axes[1, 2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "toy2d_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D toy inverse cosmology pipeline.")
    parser.add_argument("--config", type=Path, default=Path("data/toy/config.yaml"))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/toy2d"))
    args = parser.parse_args()

    config = _load_profile(args.config, args.profile)
    metrics = run_experiment(config, args.output)

    print("2D toy reconstruction complete")
    print(f"output_dir={args.output}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
