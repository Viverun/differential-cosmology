"""Gradient-based MAP inference for 2D toy reconstruction."""

from typing import Any, Dict

import jax
import jax.numpy as jnp

from diffcosmo.loss import total_loss


def reconstruct_map(
    y_obs: jnp.ndarray,
    theta_init: jnp.ndarray,
    sim_config: Dict[str, Any],
    prior_config: Dict[str, Any],
    obs_config: Dict[str, Any],
    opt_config: Dict[str, Any],
) -> Dict[str, jnp.ndarray]:
    """Optimize MAP objective over initial field using Adam."""
    learning_rate = float(opt_config.get("lr", 0.05))
    beta1 = float(opt_config.get("beta1", 0.9))
    beta2 = float(opt_config.get("beta2", 0.999))
    eps = float(opt_config.get("eps", 1e-8))
    max_iters = int(opt_config.get("max_iters", 300))
    patience = int(opt_config.get("early_stop_patience", 50))
    min_delta = float(opt_config.get("early_stop_min_delta", 1e-6))
    max_grad_norm = float(opt_config.get("max_grad_norm", 1.0))
    max_backtrack = int(opt_config.get("max_backtrack", 6))

    theta = theta_init.astype(jnp.float32)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    best_theta = theta

    objective = lambda t: total_loss(t, y_obs, sim_config, prior_config, obs_config)
    objective_and_grad = jax.jit(jax.value_and_grad(objective))
    objective_only = jax.jit(objective)

    loss_history = []
    grad_norm_history = []
    best_loss = float("inf")
    stale_steps = 0

    for step in range(1, max_iters + 1):
        loss_value, grad = objective_and_grad(theta)
        grad_norm = jnp.linalg.norm(grad)

        if (not jnp.isfinite(loss_value)) or (not jnp.isfinite(grad_norm)):
            break

        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / (grad_norm + 1e-8))
            grad_norm = max_grad_norm

        m_new = beta1 * m + (1.0 - beta1) * grad
        v_new = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m_new / (1.0 - beta1**step)
        v_hat = v_new / (1.0 - beta2**step)

        step_size = learning_rate
        accepted = False
        accepted_loss = float(loss_value)

        for _ in range(max_backtrack):
            theta_candidate = theta - step_size * m_hat / (jnp.sqrt(v_hat) + eps)
            cand_loss = objective_only(theta_candidate)
            if bool(jnp.isfinite(cand_loss)) and float(cand_loss) <= float(loss_value):
                theta = theta_candidate
                m = m_new
                v = v_new
                accepted_loss = float(cand_loss)
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            accepted_loss = float(loss_value)

        loss_scalar = accepted_loss
        grad_scalar = float(grad_norm)
        loss_history.append(loss_scalar)
        grad_norm_history.append(grad_scalar)

        if best_loss - loss_scalar > min_delta:
            best_loss = loss_scalar
            best_theta = theta
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= patience:
                break

    loss_history_arr = jnp.array(loss_history, dtype=jnp.float32)
    grad_history_arr = jnp.array(grad_norm_history, dtype=jnp.float32)
    return {
        "theta_map": best_theta,
        "loss_history": loss_history_arr,
        "grad_norm_history": grad_history_arr,
        "n_iters": jnp.array(loss_history_arr.shape[0]),
    }
