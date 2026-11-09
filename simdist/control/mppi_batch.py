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
        # design doc 15: calibrated planner score.  control.calib = {enabled, weights, tau, fall_max}: candidates are
        # scored by the tau-quantile of the REALISED 25-step return predicted by a small head (trained on the planner's
        # own traces, scripts/sim2sim/calib_head.py) instead of the model's promised return; candidates whose
        # predicted fall probability exceeds fall_max are excluded.  Everything else in MPPI is unchanged.
        self.calib_cfg = dict(controller_cfg.get("calib", {}) or {})
        self.calib = None
        if self.calib_cfg.get("enabled", False):
            w = np.load(str(self.calib_cfg["weights"]))
            self.calib = {k: jnp.asarray(w[k], jnp.float32) for k in ("w1", "b1", "w2", "b2", "w3", "b3", "mu", "sd")}
            self.calib["g_mu"] = float(w["g_mu"]); self.calib["g_sd"] = float(w["g_sd"])
            taus = np.asarray(w["taus"], np.float64)
            self.calib["tau_idx"] = int(np.argmin(np.abs(taus - float(self.calib_cfg.get("tau", 0.1)))))
            self.calib["n_hist"] = int(w["n_hist"])
            fm = self.calib_cfg.get("fall_max", None)
            self.calib["fall_max"] = None if fm in (None, "null", "none") else float(fm)

        # design doc 16: generative-prior candidates.  control.gen = {enabled, weights, dec_noise, min_std_u}: candidates are
        # decoded from a conditional VAE (scripts/sim2sim/gen_chunks.py) by CEM in its latent space instead of Gaussian noise
        # around the shifted previous plan; no policy mixture samples (B = 0); the executed plan is the decoded final mean.
        self.gen_cfg = dict(controller_cfg.get("gen", {}) or {})
        self.gen = None
        if self.gen_cfg.get("enabled", False):
            w = np.load(str(self.gen_cfg["weights"]))
            self.gen = {k: jnp.asarray(w[k], jnp.float32) for k in ("d1", "db1", "d2", "db2", "d3", "db3", "dlog_sig",
                                                                     "p_mean", "p_std", "a_mean", "a_std", "c_mean", "c_std")}
            self.gen["du"] = int(w["du"]); self.gen["dec_noise"] = float(self.gen_cfg.get("dec_noise", 0.0))
            self.gen["min_std_u"] = float(self.gen_cfg.get("min_std_u", 0.05))
            self.gen["u_max"] = float(self.gen_cfg.get("u_max", 2.5))
            self.B = 0

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
        # extra model inputs broadcast to every robot and candidate, e.g. {"context_override": (N, 3)}
        # (oracle context, design doc 13 P2)
        self.extra_inputs = {}
        # design doc 7 planner options (both default off):
        #  noise_schedule: DIAL-MPC-style two-axis annealing of the sampling noise (shrinks over
        #    iterations and toward the near end of the horizon); zero extra compute
        #  consistency_gate: if the latent predicted for this step by the previous plan is far
        #    from the encoded actual latent (> factor x running median), execute the prior's plan
        self.noise_schedule = dict(controller_cfg.get("noise_schedule", {}) or {})
        self.noise_corr = float(controller_cfg.get("noise_corr", 0.0) or 0.0)  # AR(1) coefficient of the sampling noise along T
        self.gate_cfg = dict(controller_cfg.get("consistency_gate", {}) or {})
        self.pred_next = None  # (N, latent) predicted latent for the next control step
        self.dist_hist = []  # list of (N,) distances (most recent `window`)
        self.gate_steps = 0; self.gated_steps = 0

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
        if self.pred_next is not None:
            pn = np.array(self.pred_next)
            for i in env_ids: pn[i] = np.nan  # no prediction across a reset
            self.pred_next = jnp.asarray(pn)
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
        x.update(self.extra_inputs)
        return x

    # ---- context slot (design doc 13) --------------------------------------------
    @functools.partial(nnx.jit, static_argnames=["self"])
    def _context_estimate(self, x):
        xs = self.model.scaler.scale(x)
        psi = self.model.context_from(xs["proprio_obs_hist"], xs["acts_hist"], deterministic=True)
        return self.model.unscale_xi(psi[:, :3])

    def context_estimate(self):
        """Raw [motor scale, foot friction, added mass] read by the model from every robot's
        history, shape (N, 3); None for a model without a context slot."""
        if getattr(self.model, "context", None) is None:
            return None
        return np.asarray(self._context_estimate(self._model_inputs()))

    def run_control(self) -> np.ndarray:
        """Returns the planned action sequences, shape (N, T, A)."""
        x = self._model_inputs()
        if self.external_prior is not None:
            base_actions = jnp.asarray(self.external_prior(x), jnp.float32)
        else:
            base_actions = self._base_policy_actions(x)
        if self.ctrl_cfg.get("prior_only", False):
            # execute the prior's own plan (policy head or external prior), no planning
            self.state = BatchedMppiState(base_actions.astype(jnp.float32), self.state.std, self.state.key_data)
        else:
            self.state = self._mppi_step(self.state, base_actions, x)
        actions = self.state.mean
        if self.gate_cfg.get("enabled", False):
            z_now = self._encode_now(x)
            if self.pred_next is not None:
                d = np.asarray(jnp.linalg.norm(z_now - self.pred_next, axis=-1))
                self.dist_hist.append(d); self.dist_hist = self.dist_hist[-int(self.gate_cfg.get("window", 200)):]
                if len(self.dist_hist) >= 20:
                    med = np.nanmedian(np.stack(self.dist_hist), axis=0)  # per-env running median (NaN after a reset)
                    gate = d > float(self.gate_cfg.get("factor", 3.0)) * med
                    self.gate_steps += 1; self.gated_steps += int(gate.sum()) / self.N
                    if gate.any():
                        g = jnp.asarray(gate)[:, None, None]
                        actions = jnp.where(g, base_actions, actions)
                        self.state = BatchedMppiState(actions.astype(jnp.float32), self.state.std, self.state.key_data)
            self.pred_next = self._predict_next(x, actions)
        return np.asarray(actions)

    # ---- design doc 14: trace of the model's own prediction for the executed plan ----------
    @functools.partial(nnx.jit, static_argnames=["self"])
    def _trace(self, x, actions):
        xx = dict(x); xx["fut_acts"] = actions
        y = self.model.inference(xx)
        xs = self.model.scaler.scale(x)
        z_now = self.model.encode_latent(xs["proprio_obs_hist"][:, -1], xs["extero_obs"], deterministic=True)
        return z_now, y["latents"], y["rewards"], y["values"]

    def trace(self, actions):
        """Current latent z_t (N, D) and the model's prediction for the plan that will be executed:
        latents (N, T, D), rewards (N, T), values (N, T) in raw units. Inputs = the same history
        the planner used this step (call after run_control, before env.step)."""
        z, l, r, v = self._trace(self._model_inputs(), jnp.asarray(actions, jnp.float32))
        N, T = actions.shape[0], actions.shape[1]
        return np.asarray(z), np.asarray(l).reshape(N, T, -1), np.asarray(r).reshape(N, T), np.asarray(v).reshape(N, T)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _encode_now(self, x):
        xs = self.model.scaler.scale(x)
        return self.model.encode_latent(xs["proprio_obs_hist"][:, -1], xs["extero_obs"], deterministic=True)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _predict_next(self, x, actions):
        xx = dict(x); xx["fut_acts"] = actions
        return self.model.inference(xx)["latents"][:, 0]

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

        I = int(self.ctrl_cfg["iterations"])
        if self.calib is not None:
            # history block, once per step: realised latents of the last T observations + the last T executed actions,
            # in the same layout as scripts/sim2sim/calib_head.py (trace z = encode_latent(o), trace action = executed plan[0])
            xs = self.model.scaler.scale(x)
            ph = xs["proprio_obs_hist"][:, -T:]                                        # (N, T, o_p)
            z_hist = jax.vmap(lambda p: self.model.encode_latent(p, xs["extero_obs"], deterministic=True),
                              in_axes=1, out_axes=1)(ph)                                 # (N, T, D)
            a_hist = jnp.asarray(x["acts_hist"], jnp.float32)[:, -T:]                    # (N, T, A) raw units
            hist_feat = jnp.concatenate([z_hist.reshape(N, -1), a_hist.reshape(N, -1)], -1)  # (N, n_hist)
            hist_rep = jnp.repeat(hist_feat[:, None], S + B, axis=1)                     # (N, S+B, n_hist)
            c = self.calib

            def calibrated_scores(y, returns):
                lat = y["latents"].reshape(N, S + B, T, -1)
                cand = jnp.concatenate([y["rewards"].reshape(N, S + B, T), y["values"].reshape(N, S + B, T)[..., -1:],
                                        returns[..., None], lat.mean(2), lat[:, :, -1]], -1)
                feat = (jnp.concatenate([hist_rep, cand], -1) - c["mu"]) / c["sd"]
                hh = jax.nn.relu(feat @ c["w1"] + c["b1"]); hh = jax.nn.relu(hh @ c["w2"] + c["b2"]); out = hh @ c["w3"] + c["b3"]
                score = out[..., c["tau_idx"]] * c["g_sd"] + c["g_mu"]
                if c["fall_max"] is not None:
                    score = jnp.where(jax.nn.sigmoid(out[..., 3]) > c["fall_max"], -1e6, score)
                return score
        if self.noise_schedule.get("enabled", False):
            it_f = jnp.exp(-float(self.noise_schedule.get("iter_decay", 1.0)) * jnp.arange(I) / max(I - 1, 1))  # (I,)
            h_f = jnp.exp(-float(self.noise_schedule.get("horizon_decay", 1.0)) * (T - 1 - jnp.arange(T)) / max(T - 1, 1))  # (T,) less noise near t
            sched = it_f[:, None, None] * h_f[None, :, None]  # (I, T, 1)
        else:
            sched = jnp.ones((I, T, 1))

        if self.gen is not None:
            g = self.gen; du = g["du"]
            cond = jnp.concatenate([((jnp.asarray(x["proprio_obs_hist"], jnp.float32) - g["p_mean"]) / g["p_std"]).reshape(N, -1),
                                    ((jnp.asarray(x["acts_hist"], jnp.float32) - g["a_mean"]) / g["a_std"]).reshape(N, -1),
                                    ((jnp.asarray(x["fut_cmds"], jnp.float32) - g["c_mean"]) / g["c_std"]).reshape(N, -1)], -1)  # (N, dc)
            cond_rep = jnp.repeat(cond[:, None], S, axis=1)                                                         # (N, S, dc)
            a_mean_T = jnp.tile(g["a_mean"], T); a_std_T = jnp.tile(g["a_std"], T)
            ls = jnp.clip(g["dlog_sig"], -3.0, 1.0)

            def decode(u, key):  # (N, S, du) -> raw action chunks (N, S, T, A)
                hh = jax.nn.relu(jnp.concatenate([u, cond_rep[:, :u.shape[1]]], -1) @ g["d1"] + g["db1"])
                hh = jax.nn.relu(hh @ g["d2"] + g["db2"]); m = hh @ g["d3"] + g["db3"]
                if g["dec_noise"] > 0:
                    m = m + g["dec_noise"] * jnp.exp(ls) * jax.random.normal(key, m.shape)
                return (m * a_std_T + a_mean_T).reshape(N, u.shape[1], T, A)

            def fg(carry, i):
                mu_u, sig_u, keys = carry
                keys, k_u = split(keys); keys, k_d = split(keys)
                u = mu_u[:, None] + sig_u[:, None] * jax.vmap(lambda k: jax.random.normal(k, (S, du)))(k_u)       # (N, S, du)
                u = jnp.clip(u, -g["u_max"], g["u_max"])  # stay where the decoder was trained (posterior u ~ N(0, 1))
                acts = decode(u, k_d[0])
                xx = dict(x_rep); xx["fut_acts"] = acts.reshape(N * S, T, A)
                y = self.model.inference(xx)
                rewards = jnp.sum(y["rewards"] * self.discounts, axis=-1)
                returns = (rewards + y["values"][:, -1] * self.final_discount).reshape(N, S)
                if self.calib is not None:
                    returns = calibrated_scores(y, returns)
                elite_returns, elite_idxs = jax.lax.top_k(returns, self.ctrl_cfg["num_elites"])
                elite_u = jnp.take_along_axis(u, elite_idxs[:, :, None], axis=1)                                   # (N, k, du)
                score = jnp.exp(self.ctrl_cfg["temperature"] * (elite_returns - elite_returns.max(axis=1, keepdims=True)))
                w = score / score.sum(axis=1, keepdims=True)
                new_mu = jnp.sum(w[:, :, None] * elite_u, axis=1)
                new_sig = jnp.sqrt(jnp.sum(w[:, :, None] * (elite_u - new_mu[:, None]) ** 2, axis=1))
                return (new_mu, jnp.clip(new_sig, min=g["min_std_u"]), keys), None

            (mu_u, sig_u, keys), _ = jax.lax.scan(fg, (jnp.zeros((N, du)), jnp.ones((N, du)), keys), jnp.arange(I))
            keys, k_d = split(keys)
            plan = decode(mu_u[:, None], k_d[0])[:, 0]                                                            # (N, T, A)
            return BatchedMppiState(plan.astype(jnp.float32), state.std, jax.random.key_data(keys))

        def f(carry, i):
            mean, std, keys = carry
            keys, k_act = split(keys)
            noise = jax.vmap(lambda k: jax.random.normal(k, (S, T, A)))(k_act)
            if self.noise_corr > 0.0:
                # AR(1) along the horizon: e_t = rho e_{t-1} + sqrt(1 - rho^2) w_t keeps the marginal std (design 16 supplement,
                # "coloured noise around the prior" control arm)
                rho = self.noise_corr
                w_t = jnp.moveaxis(noise, 2, 0)                                                      # (T, N, S, A)
                e_rest = jax.lax.scan(lambda e, w: (rho * e + jnp.sqrt(1.0 - rho ** 2) * w,) * 2, w_t[0], w_t[1:])[1]
                noise = jnp.moveaxis(jnp.concatenate([w_t[:1], e_rest], 0), 0, 2)                   # (N, S, T, A)
            noise = noise * (std * sched[i])[:, None]
            acts = mean[:, None] + noise if B == 0 else jnp.concatenate([mean[:, None] + noise, noised_base], axis=1)  # (N, S+B, T, A)
            xx = dict(x_rep)
            xx["fut_acts"] = acts.reshape(N * (S + B), T, A)
            y = self.model.inference(xx)
            rewards = jnp.sum(y["rewards"] * self.discounts, axis=-1)
            returns = (rewards + y["values"][:, -1] * self.final_discount).reshape(N, S + B)
            if self.calib is not None:
                returns = calibrated_scores(y, returns)

            elite_returns, elite_idxs = jax.lax.top_k(returns, self.ctrl_cfg["num_elites"])  # (N, k)
            elite_acts = jnp.take_along_axis(acts, elite_idxs[:, :, None, None], axis=1)  # (N, k, T, A)
            score = jnp.exp(self.ctrl_cfg["temperature"] * (elite_returns - elite_returns.max(axis=1, keepdims=True)))
            w = score / score.sum(axis=1, keepdims=True)  # (N, k)
            new_mean = jnp.sum(w[:, :, None, None] * elite_acts, axis=1)
            new_std = jnp.sqrt(jnp.sum(w[:, :, None, None] * (elite_acts - new_mean[:, None]) ** 2, axis=1))
            mean = self.ctrl_cfg["momentum"] * mean + (1 - self.ctrl_cfg["momentum"]) * new_mean
            std = jnp.clip(new_std, min=self.ctrl_cfg["min_std"])
            return (mean, std, keys), None

        (mean, std, keys), _ = jax.lax.scan(f, (mean, std, keys), jnp.arange(I))
        return BatchedMppiState(mean.astype(jnp.float32), std.astype(jnp.float32), jax.random.key_data(keys))
