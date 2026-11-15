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
        # design doc 24: per-episode context xi_e (auto-decoded table) for windows of an episode subset
        xi_prior = jnp.zeros(())
        xi_table = getattr(model, "xi_table", None)
        if xi_table is not None and "episode_id" in y:
            xi = xi_table.value[jnp.asarray(y["episode_id"]).reshape(-1)]
            x = dict(x)
            x["xi"] = xi
            xi_prior = jnp.mean(jnp.sum(xi**2, axis=-1))

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
        # design doc 20: simulation replay for the latent-dynamics term. ``replay=(x_sim, y_sim)`` is a batch
        # of the pretraining (simulation) windows; the term becomes (1 - w) * real + w * sim.
        replay = kwargs.get("replay")
        replay_w = float(self.loss_cfg.get("replay_weight", 0.0))
        replay_latent_loss = None
        if replay is not None and replay_w > 0.0:
            xr, yr = replay
            yr = jax.lax.stop_gradient(scaler.scale(yr))
            yr_pred = model(xr, deterministic=deterministic)
            fut_r = jax.lax.stop_gradient(model.encode_latent(yr["proprio_obs"], yr["extero_obs"], deterministic=True))
            replay_latent_loss = jnp.mean((yr_pred["latents"] - fut_r) ** 2)
        bw = float(self.loss_cfg.get("boundary_weight", 1.0))
        if (self.loss_cfg.get("prior_weight", False) and "prior_weight" in y) or (bw != 1.0 and "steps_to_end" in y):
            # per-window weights: prior-consistency (design doc 9) and/or fall-boundary weight (design doc 10 §7:
            # boundary_weight=0 drops windows that end within 25 steps of a fall)
            per_window = jnp.mean(latents_error**2, axis=(1, 2))
            w = jnp.ones_like(per_window)
            if self.loss_cfg.get("prior_weight", False) and "prior_weight" in y:
                w = w * jnp.asarray(y["prior_weight"]).reshape(-1)
            if bw != 1.0 and "steps_to_end" in y:
                is_boundary = (jnp.asarray(y["steps_to_end"])[:, -1] <= 25) & (jnp.asarray(y["terminated"])[:, -1] > 0.5)
                w = w * jnp.where(is_boundary, bw, 1.0)
            latent_dynamics_loss = jnp.sum(w * per_window) / (jnp.sum(w) + 1e-8)
        else:
            latent_dynamics_loss = jnp.mean(latents_error**2)
        if replay_latent_loss is not None:
            latent_dynamics_loss = (1.0 - replay_w) * latent_dynamics_loss + replay_w * replay_latent_loss

        reward_error = y_pred["rewards"] - y["rewards"]
        reward_loss = jnp.mean(reward_error**2)

        value_error = y_pred["values"] - y["values"]
        value_loss = jnp.mean(value_error**2)

        # set loss to zero for all actions after the expert policy isn't used
        act_loss_mask = jnp.cumprod(y["exp_pol_flags"].astype(jnp.int32), axis=-1)
        act_loss_mask = act_loss_mask[:, :, None]

        action_error = y_pred["actions"] - y["actions"]
        action_loss = jnp.mean((action_error * act_loss_mask) ** 2)

        # context slot (design doc 13): psi[:, :3] regresses the standardized physics label of the
        # window (labelled windows only), and psi from the history must agree with psi from the
        # future segment of the same episode (symmetric, each side stop-gradient). Zero without a slot.
        psi = y_pred.get("context")
        if psi is not None and "physics" in y:
            x_s = jax.lax.stop_gradient(scaler.scale(x))
            valid = jnp.asarray(y["physics_valid"], jnp.float32).reshape(-1)
            xi_s = jax.lax.stop_gradient(model.scale_xi(jnp.asarray(y["physics"])))
            per_window = jnp.mean((psi[:, :3] - xi_s) ** 2, axis=-1)
            context_loss = jnp.sum(valid * per_window) / (jnp.sum(valid) + 1e-8)
            ph_fut = jnp.concatenate([x_s["proprio_obs_hist"][:, -1:], y["proprio_obs"]], axis=1)  # o_t .. o_{t+T}
            psi_fut = model.context_from(ph_fut, x_s["fut_acts"], deterministic=deterministic)
            context_consistency = jnp.mean((psi - jax.lax.stop_gradient(psi_fut)) ** 2) + jnp.mean(
                (psi_fut - jax.lax.stop_gradient(psi)) ** 2
            )
        else:
            context_loss = jnp.zeros(())
            context_consistency = jnp.zeros(())

        losses = {
            "latent_dynamics": latent_dynamics_loss,
            "reward": reward_loss,
            "value": value_loss,
            "action": action_loss,
            "context": context_loss,
            "context_consistency": context_consistency,
            "xi_prior": xi_prior,
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
        # "anchor": bootstrap with the frozen pretrained value (anchor head if the
        #           model has one, otherwise the head without the residual)
        # "self":   bootstrap with the value being adapted (TD-style), which lets
        #           the value scale move away from the simulation value
        self.bootstrap = self.loss_cfg.get("bootstrap", "anchor")
        assert self.bootstrap in ("anchor", "self")
        # "labels": rewards logged with the data (simulation / sim-to-sim logs)
        # "model":  rewards predicted by the frozen reward head on the encoded
        #           true latents (real-world data carries no reward)
        self.reward_source = self.loss_cfg.get("reward_source", "labels")
        assert self.reward_source in ("labels", "model")
        # terminal-aware bootstrap: when the episode ends by a fall within
        # ``max_steps`` after the window, bootstrap with the logged discounted
        # reward-to-go (zero after the fall) instead of the anchor, so the value
        # targets carry the shifted failure boundary (docs/design_value_adaptation_ko.md)
        term_cfg = self.loss_cfg.get("terminal", {}) or {}
        self.terminal_enabled = bool(term_cfg.get("enabled", False))
        self.terminal_max_steps = float(term_cfg.get("max_steps", 50))
        # decision-aware dynamics adaptation (docs/design_dynamics_adaptation_ko.md):
        #   ret:             weight "ret" in weights -> return-matching term: the planner's
        #                    return of the executed actions on predicted latents must match
        #                    the return on the true latents (reward and value heads frozen)
        #   boundary_weight: latent-dynamics loss of windows that end within max_steps of a
        #                    fall is multiplied by this factor (ablation: attention only)
        self.boundary_weight = float(self.loss_cfg.get("boundary_weight", 1.0))

    def compute_losses(
        self,
        model: models.WorldModelBase,
        x: types.WorldModelSchema.Inputs,
        y: types.WorldModelSchema.Labels,
        deterministic: bool,
        replay: tuple | None = None,
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
        per_window = jnp.mean((y_pred["latents"] - fut_latents) ** 2, axis=(1, 2))  # (B,)
        if self.boundary_weight != 1.0 and "steps_to_end" in y:
            is_boundary = (jnp.asarray(y["steps_to_end"])[:, -1] <= self.terminal_max_steps) & (
                jnp.asarray(y["terminated"])[:, -1] > 0.5
            )
            w = jnp.where(is_boundary, self.boundary_weight, 1.0)
            latent_dynamics_loss = jnp.sum(w * per_window) / jnp.sum(w)
        else:
            latent_dynamics_loss = jnp.mean(per_window)

        # bootstrap value of the last true latent
        fut_cmds_next = x_s["fut_cmds"][:, 1:]
        if self.bootstrap == "self":
            v_boot_s = model.value_from_latents(
                fut_latents, fut_cmds_next, deterministic=True, use_residual=True
            )
        else:
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
        if self.terminal_enabled and "steps_to_end" in y:
            near_end = jnp.asarray(y["steps_to_end"])[:, -1] <= self.terminal_max_steps
            fell = jnp.asarray(y["terminated"])[:, -1] > 0.5
            v_boot = jnp.where(near_end & fell, jnp.asarray(y["rtg"])[:, -1], v_boot)

        # discounted returns G_j from step t+j (reward r_j onwards), j = 0..T-1
        if self.reward_source == "model":
            rp = scaler.get_scaler_params()["rewards"]
            rewards_s = model.reward_from_latents(x_s, fut_latents, deterministic=True)
            rewards = jax.lax.stop_gradient(rewards_s * (rp["std"] + 1e-8) + rp["mean"])
        else:
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
            delta = model.value_residual(
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

        # return-matching: R_hat on predicted latents vs R_true on true latents (scaled
        # units; the value scale normalizes the term). Gradients reach the dynamics (and
        # the residual through V(z_hat)); reward / value heads are frozen by the head config.
        if "ret" in self.weights:
            T = fut_latents.shape[1]
            disc = self.discount ** jnp.arange(T)
            final = self.discount ** (T + 1)
            rp = scaler.get_scaler_params()["rewards"]
            r_scale = (rp["std"][0] + 1e-8) / v_std
            r_hat = y_pred["rewards"]
            v_hat = y_pred["values"]
            R_hat = jnp.sum(r_hat * disc, axis=-1) * r_scale + final * v_hat[:, -1]
            r_true = jax.lax.stop_gradient(model.reward_from_latents(x_s, fut_latents, deterministic=True))
            v_true = jax.lax.stop_gradient(
                model.value_from_latents(fut_latents, fut_cmds_next, deterministic=True, use_residual=True)
            )
            R_true = jnp.sum(r_true * disc, axis=-1) * r_scale + final * v_true[:, -1]
            losses["ret"] = jnp.mean((R_hat - R_true) ** 2)

        # optional data anchor: on windows of the simulation dataset the adapted
        # value must still match the simulation critic's labels
        if "value_replay" in self.weights:
            if replay is not None:
                xr, yr = replay
                xr_s = jax.lax.stop_gradient(scaler.scale(xr))
                yr_s = jax.lax.stop_gradient(scaler.scale(yr))
                lat_r = jax.lax.stop_gradient(
                    model.encode_latent(yr_s["proprio_obs"], yr_s["extero_obs"], deterministic=True)
                )
                v_r = model.value_from_latents(
                    lat_r, xr_s["fut_cmds"][:, 1:], deterministic=deterministic, use_residual=True
                )
                losses["value_replay"] = jnp.mean((v_r - yr_s["values"]) ** 2)
            else:
                losses["value_replay"] = jnp.zeros(())
        return losses, {}


@register_loss("head_refit")
class HeadRefitLoss(Loss):
    """Design doc 22: refit the reward and value heads of an adapted model on logged data (dynamics frozen
    by the head config).

    ``value_input``: "pred" - the heads read the latents predicted by the (adapted) dynamics, the same path
    as pretraining and planning; "enc" - the heads read the encoded true future latents (design doc 1).
    Value labels: logged discounted reward-to-go (zero after a fall); steps of episodes that were cut by the
    end of the run within ``trunc_mask`` steps are masked (their reward-to-go is truncated).
    Reward labels: logged rewards.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.value_input = self.loss_cfg.get("value_input", "pred")
        assert self.value_input in ("pred", "enc")
        self.trunc_mask = float(self.loss_cfg.get("trunc_mask", 300))
        # design doc 22 §13 conservative term (weight "cons"): K perturbed candidates per window (raw action noise
        # with the stds below, like the MPPI samples); the refit may not raise a candidate's planner score J relative
        # to the logged actions by more than the frozen heads (anchors, adapt.head_anchor=true) did:
        #   cons = mean( relu( [J(c) - J(log)] - [J_anchor(c) - J_anchor(log)] )^2 ) / std_value^2
        cons = self.loss_cfg.get("cons", {}) or {}
        self.cons_K = int(cons.get("K", 16))
        self.cons_sigmas = [float(v) for v in cons.get("sigmas", [0.1, 0.3, 1.0, 2.0])]
        self.discount = float(self.loss_cfg.get("discount", 0.99))

    def compute_losses(self, model, x, y, deterministic: bool, **kwargs):
        scaler = model.get_scaler()
        y_s = jax.lax.stop_gradient(scaler.scale(y))
        if self.value_input == "pred":
            y_pred = model(x, deterministic=deterministic)
            v_s, r_s = y_pred["values"], y_pred["rewards"]
        else:
            x_s = jax.lax.stop_gradient(scaler.scale(x))
            fut = jax.lax.stop_gradient(model.encode_latent(y_s["proprio_obs"], y_s["extero_obs"], deterministic=True))
            v_s = model.value_from_latents(fut, x_s["fut_cmds"][:, 1:], deterministic=deterministic, use_residual=True)
            r_s = model.reward_from_latents(x_s, fut, deterministic=deterministic)
        vp = scaler.get_scaler_params()["values"]
        target = jax.lax.stop_gradient((jnp.asarray(y["rtg"]) - vp["mean"]) / (vp["std"] + 1e-8))
        valid = (jnp.asarray(y["terminated"]) > 0.5) | (jnp.asarray(y["steps_to_end"]) >= self.trunc_mask)
        w = valid.astype(jnp.float32)
        value_loss = jnp.sum(w * (v_s - target) ** 2) / (jnp.sum(w) + 1e-8)
        reward_loss = jnp.mean((r_s - y_s["rewards"]) ** 2)
        losses = {"reward": reward_loss, "value": value_loss}
        if "cons" in self.weights:
            losses["cons"] = self._conservative(model, x, scaler) if float(self.weights["cons"]) > 0 else jnp.zeros(())
        return losses, {}

    def _conservative(self, model, x, scaler):
        assert model.value_anchor is not None and model.reward_anchor is not None, "cons needs adapt.head_anchor=true"
        B, T, A = x["fut_acts"].shape
        K = self.cons_K
        # a data-dependent key (losses get no rng); different batches give different candidates
        seed = jnp.mod(jnp.sum(jnp.abs(x["fut_acts"])) * 1000.0, 2.0 ** 30).astype(jnp.int32)
        key = jax.random.fold_in(jax.random.PRNGKey(22), seed)
        sig = jnp.asarray([self.cons_sigmas[k % len(self.cons_sigmas)] for k in range(K)], jnp.float32)
        noise = jax.random.normal(key, (K, B, T, A)) * sig[:, None, None, None]
        xc = {k: jnp.concatenate([v[None], jnp.broadcast_to(v[None], (K,) + v.shape)], 0).reshape(((K + 1) * B,) + v.shape[1:])
              for k, v in x.items()}
        acts = jnp.concatenate([x["fut_acts"][None], x["fut_acts"][None] + noise], 0)  # index 0 = logged
        xc["fut_acts"] = acts.reshape((K + 1) * B, T, A)
        out = model(xc, deterministic=True)
        xc_s = jax.lax.stop_gradient(scaler.scale(xc))
        lat = jax.lax.stop_gradient(out["latents"])
        r_old = model.reward_from_latents(xc_s, lat, deterministic=True, use_anchor=True)
        v_old = model.value_from_latents(lat, xc_s["fut_cmds"][:, 1:], deterministic=True, use_residual=False, use_anchor=True)
        sp = scaler.get_scaler_params()
        r_std, v_std = sp["rewards"]["std"].reshape(()) + 1e-8, sp["values"]["std"].reshape(()) + 1e-8
        disc = self.discount ** jnp.arange(T)

        def J(r, v):  # raw-unit planner score up to a constant (means cancel in differences)
            return (jnp.sum(r * disc, -1) * r_std + v[:, -1] * v_std * self.discount ** (T + 1)).reshape(K + 1, B)

        j_new = J(out["rewards"], out["values"])
        j_old = jax.lax.stop_gradient(J(r_old, v_old))
        excess = (j_new[1:] - j_new[:1]) - (j_old[1:] - j_old[:1])
        return jnp.mean(jax.nn.relu(excess) ** 2) / (v_std ** 2)


@register_loss("policy_refresh")
class PolicyRefreshLoss(Loss):
    """Re-optimize the base policy (planner prior) in imagination with the adapted dynamics.

    For each window the policy produces an action sequence from the real history; the
    (adapted) latent dynamics rolls it out; the frozen reward and value heads score it:
        R_hat = sum_k gamma^k r_k + gamma^(T+1) V(z_T)   (scaled by the value std)
    The loss maximizes R_hat (term "policy_return" = -R_hat) and keeps the policy close
    to its frozen simulation copy (term "policy_anchor" = mean squared action
    difference, in scaled action units). Only the policy heads should be trainable
    (heads=policy_only). See docs/design_prior_refresh_ko.md, section 4.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.discount = float(self.loss_cfg.get("discount", 0.99))

    def compute_losses(self, model, x, y, deterministic: bool, **kwargs):
        out = model.imagine_with_policy(x, deterministic=deterministic)
        scaler = model.get_scaler()
        sp = scaler.get_scaler_params()
        v_std = sp["values"]["std"][0] + 1e-8
        r_scale = (sp["rewards"]["std"][0] + 1e-8) / v_std
        T = out["latents"].shape[1]
        disc = self.discount ** jnp.arange(T)
        R_hat = jnp.sum(out["rewards"] * disc, axis=-1) * r_scale + (self.discount ** (T + 1)) * out["values"][:, -1]
        a_sim = jax.lax.stop_gradient(model.imagine_with_policy(x, deterministic=True, use_anchor=True)["actions"])
        anchor = jnp.mean((out["actions"] - a_sim) ** 2)
        return {"policy_return": -jnp.mean(R_hat), "policy_anchor": anchor}, {}
