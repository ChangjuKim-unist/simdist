"""Regression tests for the fixes of the upstream audit (docs/audit_upstream_ko.md, section 5)."""

import flax.nnx as nnx
import jax.numpy as jnp
import yaml

from simdist.data.episode_logger import fell_before_limit
from simdist.utils import model as model_utils, paths


class _Head(nnx.Module):
    def __init__(self, rngs):
        self.policy = nnx.Linear(4, 4, rngs=rngs)


class _Toy(nnx.Module):
    """Mirrors the attribute layout of WorldModelBase that matters for the heads filter."""

    def __init__(self):
        rngs = nnx.Rngs(0)
        self.dynamics = nnx.Linear(4, 4, rngs=rngs)
        self.policy = nnx.Linear(4, 4, rngs=rngs)
        self.policy_dec = nnx.Linear(4, 2, rngs=rngs)
        self.prior_anchor = _Head(rngs)
        self.value = nnx.Linear(4, 4, rngs=rngs)
        self.value_dec = nnx.Linear(4, 1, rngs=rngs)
        self.value_res = nnx.Linear(4, 1, rngs=rngs)
        self.dyn_res = nnx.Linear(4, 4, rngs=rngs)


def _selected(model, heads):
    state = nnx.state(model).filter(model_utils.create_param_filter(heads))
    return {path[0] for path, _ in nnx.to_flat_state(state)}


def test_heads_filter_selects_top_level_modules_only():
    m = _Toy()
    assert _selected(m, ["policy", "policy_dec"]) == {"policy", "policy_dec"}
    # the frozen anchor copy (prior_anchor.policy) must not be trainable under the policy heads
    assert "prior_anchor" not in _selected(m, ["policy"])
    # "dynamics" must not pick up the residual dynamics head, "value" not value_dec / value_res
    assert _selected(m, ["dynamics"]) == {"dynamics"}
    assert _selected(m, ["value"]) == {"value"}
    assert _selected(m, ["value_res"]) == {"value_res"}


def test_policy_only_heads_config_includes_the_decoder():
    heads = yaml.safe_load(open(paths.get_config_dir() + "/heads/policy_only.yaml"))
    assert "policy" in heads and "policy_dec" in heads


def test_fell_before_limit_uses_the_given_episode_length():
    limit = 5000
    assert fell_before_limit(6, limit)
    assert fell_before_limit(4990, limit)
    assert not fell_before_limit(4995, limit)
    assert not fell_before_limit(5000, limit)
