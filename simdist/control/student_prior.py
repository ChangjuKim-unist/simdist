"""History-conditioned action-sequence prior ("student") for the sampling-based planner.

A small MLP that maps the planner's own inputs -- the last H+1 proprioceptive
observations, the last H actions and the T+1 future commands -- to a T-step action
sequence. It is trained by behavior cloning on expert data (scripts/train_student_prior.py)
and replaces the world model's policy head as the source of MPPI's prior samples, or acts
alone (receding horizon). No privileged information: the dynamics regime must be read
from the history. See docs/design_prior_coverage_ko.md.
"""

import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from simdist.utils import paths


class StudentMLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, rngs: nnx.Rngs):
        dims = [in_dim] + list(hidden)
        self.layers = [nnx.Linear(a, b, rngs=rngs) for a, b in zip(dims[:-1], dims[1:])]
        self.out = nnx.Linear(dims[-1], out_dim, rngs=rngs)

    def __call__(self, x):
        for l in self.layers:
            x = jax.nn.gelu(l(x))
        return self.out(x)


def students_dir():
    return os.path.join(paths.get_model_checkpoints_dir(), "..", "students")


class StudentPrior:
    """Loads a trained student and evaluates it on raw (unscaled) planner inputs."""

    def __init__(self, name: str):
        d = os.path.join(students_dir(), name)
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.scaler = json.load(open(os.path.join(d, "scaler_params.json")))
        self.H, self.T, self.A = self.cfg["H"], self.cfg["T"], self.cfg["act_dim"]
        self.model = StudentMLP(self.cfg["in_dim"], self.cfg["hidden"], self.T * self.A, nnx.Rngs(0))
        graphdef, state = nnx.split(self.model)
        state.replace_by_pure_dict(pickle.load(open(os.path.join(d, "params.pkl"), "rb")))
        self.model = nnx.merge(graphdef, state)
        self._sc = {k: (jnp.asarray(v["mean"]), jnp.asarray(v["std"]) + 1e-8) for k, v in self.scaler.items()}

        @nnx.jit
        def fwd(model, x):
            return model(x)

        self._fwd = fwd

    @staticmethod
    def features(x, sc):
        """Scaled, flattened inputs (N, in_dim) from a planner input dict."""
        p = (x["proprio_obs_hist"] - sc["proprio_obs"][0]) / sc["proprio_obs"][1]
        a = (x["acts_hist"] - sc["actions"][0]) / sc["actions"][1]
        c = (x["fut_cmds"] - sc["commands"][0]) / sc["commands"][1]
        N = p.shape[0]
        return jnp.concatenate([p.reshape(N, -1), a.reshape(N, -1), c.reshape(N, -1)], axis=-1)

    def __call__(self, x) -> jnp.ndarray:
        """Raw action sequences (N, T, A) for raw planner inputs."""
        f = self.features({k: jnp.asarray(v) for k, v in x.items()}, self._sc)
        out = self._fwd(self.model, f).reshape(f.shape[0], self.T, self.A)
        return out * self._sc["actions"][1] + self._sc["actions"][0]
