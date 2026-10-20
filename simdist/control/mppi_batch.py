"""MPPI for several simulated robots at once.

Same algorithm and hyper-parameters as :class:`simdist.control.mppi.MppiController`,
but the controller keeps the history buffers, future commands and MPPI state of
``num_envs`` robots and evaluates the world model on all candidates of all robots
in one batch, so that one Isaac Sim instance can evaluate many seeds / speeds in
parallel. Each robot has its own random key (seed + env index), so the rollouts are
independent exactly as separate single-robot runs with different ``control.seed``.
"""

import functools

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from simdist.modeling import models, types
from simdist.utils import config


@flax.struct.dataclass
class BatchedMppiState:
    mean: jnp.ndarray  # (N, T, A)
    std: jnp.ndarray  # (N, T, A)
    key_data: jnp.ndarray  # (N, ...) raw key data


class BatchedMppiController:
    def __init__(
        self,
        model: models.WorldModelBase,
        model_cfg: dict,
        controller_cfg: dict,
        num_envs: int,
        seeds: list[int],
    ):
        self.model = model
        self.ctrl_cfg = controller_cfg
        self.N = int(num_envs)
        assert len(seeds) == self.N
        self.seeds = [int(s) for s in seeds]
        sys_cfg = model_cfg["system"]
        self.H = config.history_length_from_config(model_cfg)
        self.T = config.prediction_length_from_config(model_cfg)
        self.act_dim = config.action_dim_from_sys_config(sys_cfg)
        self.proprio_dim = config.proprio_obs_dim_from_sys_config(sys_cfg)
        self.extero_dim = config.extero_obs_dim_from_sys_config(sys_cfg)
        self.cmd_dim = config.cmd_dim_from_sys_config(sys_cfg)
        self.S = int(controller_cfg["num_samples"])
        self.B = int(controller_cfg["num_samples"] * controller_cfg["mixture_coef"])  # may be 0 (no prior samples)
        # how the sampling mean is initialized every step (design doc 3, section 3):
        #   shift    - previous plan shifted by `delay` (SimDist default, warm start)
        #   zero     - zeros (no warm start)
        #   prior    - the base policy's action sequence
        #   external - whatever state.mean holds (set by the caller, e.g. expert action)
        self.init_mode = str(controller_cfg.get("init_mode", "shift"))
        assert self.init_mode in ("shift", "zero", "prior", "external")
        self.shift = int(controller_cfg["delay"])
        self.discounts = controller_cfg["discount"] ** jnp.arange(self.T)
        self.final_discount = controller_cfg["discount"] ** (self.T + 1)
        self.dummy_actions = jnp.zeros((self.N, self.T, self.act_dim))

        # history buffers (oldest -> newest along axis 1)
        self.proprio_hist = np.zeros((self.N, self.H + 1, self.proprio_dim), np.float32)
        self.acts_hist = np.zeros((self.N, self.H, self.act_dim), np.float32)
        self.extero = np.zeros((self.N, self.extero_dim), np.float32)
        self.fut_cmds = np.zeros((self.N, self.T + 1, self.cmd_dim), np.float32)
        self.reset_counts = np.zeros(self.N, np.int64)
        self.state = None
        # optional external prior (e.g. a history-conditioned student, design doc 4):
        # callable raw planner inputs -> raw action sequences (N, T, A); replaces the
        # world model's policy head as the source of the prior samples
        self.external_prior = None

    # ---- buffers -------------------------------------------------------------
    def fill(self, env_ids, proprio_obs, extero_obs, prev_action):
        """Fill the history of the given robots with the current measurement."""
        for i in env_ids:
            self.proprio_hist[i, :] = proprio_obs[i]
            self.acts_hist[i, :] = prev_action[i]
            self.extero[i] = extero_obs[i]

    def update(self, proprio_obs, extero_obs, prev_action):
        """Append one measurement for every robot (arrays of shape (N, ...))."""
        self.proprio_hist = np.roll(self.proprio_hist, -1, axis=1)
        self.proprio_hist[:, -1] = proprio_obs
        self.acts_hist = np.roll(self.acts_hist, -1, axis=1)
        self.acts_hist[:, -1] = prev_action
        self.extero[:] = extero_obs

    def set_fut_cmd(self, cmd):
        """cmd: (N, cmd_dim), held constant over the horizon."""
        self.fut_cmds[:] = np.asarray(cmd, np.float32)[:, None, :]

    # ---- MPPI state ------------------------------------------------------------
    def _env_key_data(self, i: int):
        key = jax.random.fold_in(jax.random.key(self.seeds[i]), int(self.reset_counts[i]))
        return np.asarray(jax.random.key_data(key))

    def reset(self, env_ids, proprio_obs, extero_obs, prev_action, cmd):
        """Reset the MPPI state and history of the given robots."""
        self.fill(env_ids, proprio_obs, extero_obs, prev_action)
        self.set_fut_cmd(cmd)
        if self.state is None:
            key_data = np.stack([self._env_key_data(i) for i in range(self.N)])
            self.state = BatchedMppiState(
                mean=jnp.zeros((self.N, self.T, self.act_dim), jnp.float32),
                std=jnp.full((self.N, self.T, self.act_dim), self.ctrl_cfg["init_std"], jnp.float32),
                key_data=jnp.asarray(key_data),
            )
            return
        # np.asarray of a jax array is read-only: copy before editing
        mean = np.array(self.state.mean); std = np.array(self.state.std)
        key_data = np.array(self.state.key_data)
        for i in env_ids:
            self.reset_counts[i] += 1
            mean[i] = 0.0
            std[i] = self.ctrl_cfg["init_std"]
            key_data[i] = self._env_key_data(i)
        self.state = BatchedMppiState(jnp.asarray(mean), jnp.asarray(std), jnp.asarray(key_data))

    # ---- control ----------------------------------------------------------------
    def _model_inputs(self):
        x: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": jnp.asarray(self.proprio_hist),
            "extero_obs": jnp.asarray(self.extero),
            "acts_hist": jnp.asarray(self.acts_hist),
            "fut_acts": self.dummy_actions,
            "fut_cmds": jnp.asarray(self.fut_cmds),
        }
        return x

    def run_control(self) -> np.ndarray:
        """Returns the planned action sequences, shape (N, T, A)."""
        x = self._model_inputs()
        if self.external_prior is not None:
            base_actions = jnp.asarray(self.external_prior(x), jnp.float32)
        else:
            base_actions = self._base_policy_actions(x)
        self.state = self._mppi_step(self.state, base_actions, x)
        return np.asarray(self.state.mean)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _base_policy_actions(self, x):
        return self.model.inference(x)["actions"]  # (N, T, A)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _mppi_step(self, state: BatchedMppiState, base_actions, x) -> BatchedMppiState:
        N, S, B, T, A = self.N, self.S, self.B, self.T, self.act_dim
        keys = jax.random.wrap_key_data(state.key_data)  # (N,)

        def split(keys):
            k = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
            return k[:, 0], k[:, 1]

        keys, k_base = split(keys)
        base_noise = jax.vmap(lambda k: jax.random.normal(k, (B, T, A)))(k_base) * self.ctrl_cfg["base_act_std"]
        noised_base = base_actions[:, None] + base_noise  # (N, B, T, A)

        if self.init_mode == "shift":
            mean = jnp.roll(state.mean, shift=self.shift, axis=1)
            mean = mean.at[:, self.shift :].set(0.0)
        elif self.init_mode == "zero":
            mean = jnp.zeros_like(state.mean)
        elif self.init_mode == "prior":
            mean = base_actions
        else:  # external
            mean = state.mean
        std = jnp.full((N, T, A), self.ctrl_cfg["init_std"])

        # every env's inputs repeated for its S + B candidates
        x_rep = jax.tree.map(lambda a: jnp.repeat(a, S + B, axis=0), x)

        def f(carry, _):
            mean, std, keys = carry
            keys, k_act = split(keys)
            noise = jax.vmap(lambda k: jax.random.normal(k, (S, T, A)))(k_act) * std[:, None]
            acts = mean[:, None] + noise if B == 0 else jnp.concatenate([mean[:, None] + noise, noised_base], axis=1)  # (N, S+B, T, A)
            xx = dict(x_rep)
            xx["fut_acts"] = acts.reshape(N * (S + B), T, A)
            y = self.model.inference(xx)
            rewards = jnp.sum(y["rewards"] * self.discounts, axis=-1)
            returns = (rewards + y["values"][:, -1] * self.final_discount).reshape(N, S + B)

            elite_returns, elite_idxs = jax.lax.top_k(returns, self.ctrl_cfg["num_elites"])  # (N, k)
            elite_acts = jnp.take_along_axis(acts, elite_idxs[:, :, None, None], axis=1)  # (N, k, T, A)
            score = jnp.exp(self.ctrl_cfg["temperature"] * (elite_returns - elite_returns.max(axis=1, keepdims=True)))
            w = score / score.sum(axis=1, keepdims=True)  # (N, k)
            new_mean = jnp.sum(w[:, :, None, None] * elite_acts, axis=1)
            new_std = jnp.sqrt(jnp.sum(w[:, :, None, None] * (elite_acts - new_mean[:, None]) ** 2, axis=1))
            mean = self.ctrl_cfg["momentum"] * mean + (1 - self.ctrl_cfg["momentum"]) * new_mean
            std = jnp.clip(new_std, min=self.ctrl_cfg["min_std"])
            return (mean, std, keys), None

        (mean, std, keys), _ = jax.lax.scan(f, (mean, std, keys), jnp.arange(self.ctrl_cfg["iterations"]))
        return BatchedMppiState(mean.astype(jnp.float32), std.astype(jnp.float32), jax.random.key_data(keys))
