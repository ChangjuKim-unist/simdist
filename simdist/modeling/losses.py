from typing import Tuple, Any

import flax.nnx as nnx
import jax.numpy as jnp
import jax

from simdist.modeling import types, models
from simdist.utils import registry


Losses = dict[str, jnp.ndarray]
ExtraInfo = dict[str, Any]
_LOSS_REGISTRY: registry.Registry["Loss"] = registry.Registry("Loss")


def register_loss(name: str):
    return _LOSS_REGISTRY.register(name)


def get_loss(cfg: dict) -> "Loss":
    loss_name = cfg["loss"]["type"]
    return _LOSS_REGISTRY.create(loss_name, cfg)


class Loss:
    def __init__(self, cfg: dict):
        self.loss_cfg: dict = cfg["loss"]
        self.weights: dict = self.loss_cfg["weights"]

    @property
    def loss_terms(self) -> list[str]:
        return list(self.weights.keys())

    def __call__(
        self,
        model: nnx.Module,
        x: types.ModelInputs,
        y: types.TrainingLabels,
        deterministic: bool,
        aux_losses: dict = {},
        **kwargs,
    ) -> Tuple[jnp.ndarray, Losses]:
        losses, _ = self.compute_losses(model, x, y, deterministic, **kwargs)
        losses.update(aux_losses)
        assert set(losses.keys()) == set(self.loss_terms), (
            "Losses do not match weights. "
            f"Got {losses.keys()} losses and {self.weights.keys()} weights. "
            "Check the loss config file and add or remove weight terms as needed."
        )
        weighted_losses = {k: self.weights[k] * losses[k] for k in self.loss_terms}
        loss = sum(weighted_losses.values())
        return loss, weighted_losses

    def compute_losses(
        self,
        model: nnx.Module,
        x: types.ModelInputs,
        y: types.TrainingLabels,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[Losses, ExtraInfo]:
        """Returns a dict of losses (shape (B,)) with keys matching the weights in
        the config file, and a dict of extra info"""
        raise NotImplementedError("Must implement compute_losses method")


@register_loss("world_model")
class WorldModelLoss(Loss):
    def compute_losses(
        self,
        model: models.WorldModelBase,
        x: types.WorldModelSchema.Inputs,
        y: types.WorldModelSchema.Labels,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[Losses, ExtraInfo]:
        # Compute the predicted outputs
        y_pred = model(x, deterministic=deterministic)

        # Scale the training labels to automatically scale the losses
        scaler = model.get_scaler()
        y = jax.lax.stop_gradient(scaler.scale(y))

        # Encode future obs for latent dynamics loss
        fut_latents = jax.lax.stop_gradient(
            model.encode_latent(y["proprio_obs"], y["extero_obs"], deterministic=True)
        )

        latents_error = y_pred["latents"] - fut_latents
        latent_dynamics_loss = jnp.mean(latents_error**2)

        reward_error = y_pred["rewards"] - y["rewards"]
        reward_loss = jnp.mean(reward_error**2)

        value_error = y_pred["values"] - y["values"]
        value_loss = jnp.mean(value_error**2)

        # set loss to zero for all actions after the expert policy isn't used
        act_loss_mask = jnp.cumprod(y["exp_pol_flags"].astype(jnp.int32), axis=-1)
        act_loss_mask = act_loss_mask[:, :, None]

        action_error = y_pred["actions"] - y["actions"]
        action_loss = jnp.mean((action_error * act_loss_mask) ** 2)

        losses = {
            "latent_dynamics": latent_dynamics_loss,
            "reward": reward_loss,
            "value": value_loss,
            "action": action_loss,
        }

        return losses, {}


@register_loss("world_model_value_adapt")
class WorldModelValueAdaptLoss(Loss):
    """Latent dynamics loss plus selective value adaptation on real-world data.

    Real-world logs carry rewards (from the simulator in sim-to-sim experiments, or
    from a transferred reward model) but no value labels, so value targets are
    H-step returns of the logged rewards bootstrapped with the pretrained
    simulation value at the end of the window (SGFT-style truncated targets):

        G_k = sum_{j=k}^{T-1} gamma^(j-k) r_j + gamma^(T-k) V_sim(z_T)

    The bootstrap uses the frozen anchor head when the model has one (so a head
    that is being trained does not chase its own targets), otherwise the current
    value head. Values are evaluated on the encoded true future latents, so the
    value update does not depend on the accuracy of the adapted dynamics.
    ``value_reg`` is an L2 penalty on the residual head's output (zero when the
    model has no residual head).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.discount = float(self.loss_cfg.get("discount", 0.99))

    def compute_losses(
        self,
        model: models.WorldModelBase,
        x: types.WorldModelSchema.Inputs,
        y: types.WorldModelSchema.Labels,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[Losses, ExtraInfo]:
        y_pred = model(x, deterministic=deterministic)
        scaler = model.get_scaler()
        y_s = jax.lax.stop_gradient(scaler.scale(y))
        x_s = jax.lax.stop_gradient(scaler.scale(x))

        # latent dynamics loss (as in WorldModelLoss)
        fut_latents = jax.lax.stop_gradient(
            model.encode_latent(y_s["proprio_obs"], y_s["extero_obs"], deterministic=True)
        )
        latent_dynamics_loss = jnp.mean((y_pred["latents"] - fut_latents) ** 2)

        # bootstrap value of the last true latent, from the frozen pretrained head
        fut_cmds_next = x_s["fut_cmds"][:, 1:]
        v_boot_s = model.value_from_latents(
            fut_latents,
            fut_cmds_next,
            deterministic=True,
            use_residual=False,
            use_anchor=model.value_anchor is not None,
        )
        v_boot_s = jax.lax.stop_gradient(v_boot_s)
        vp = scaler.get_scaler_params()["values"]
        v_mean, v_std = vp["mean"], vp["std"] + 1e-8
        v_boot = v_boot_s[:, -1] * v_std + v_mean  # (B,), raw units

        # discounted returns G_j from step t+j (reward r_j onwards), j = 0..T-1
        rewards = jnp.asarray(y["rewards"])  # raw units, (B, T)
        gamma = self.discount

        def _accumulate(carry, r_j):
            g = r_j + gamma * carry
            return g, g

        _, g_rev = jax.lax.scan(_accumulate, v_boot, jnp.swapaxes(rewards, 0, 1)[::-1])
        returns = jnp.swapaxes(g_rev[::-1], 0, 1)  # (B, T), returns[:, j] = G_j
        # value label k (k = 1..T) is the state at t+k: G_k for k < T, bootstrap at k = T
        targets = jnp.concatenate([returns[:, 1:], v_boot[:, None]], axis=-1)
        targets_s = jax.lax.stop_gradient((targets - v_mean) / v_std)

        v_pred_s = model.value_from_latents(
            fut_latents, fut_cmds_next, deterministic=deterministic, use_residual=True
        )
        value_adapt_loss = jnp.mean((v_pred_s - targets_s) ** 2)

        if model.value_res is not None:
            delta = model.value_res(
                jnp.concatenate((fut_latents, fut_cmds_next), axis=-1),
                deterministic=deterministic,
            ).squeeze(-1)
            value_reg = jnp.mean(delta**2)
        else:
            value_reg = jnp.zeros(())

        losses = {
            "latent_dynamics": latent_dynamics_loss,
            "value_adapt": value_adapt_loss,
            "value_reg": value_reg,
        }
        return losses, {}
