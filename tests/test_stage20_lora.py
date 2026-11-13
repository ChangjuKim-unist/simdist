"""Design doc 20: low-rank adapters and parameter masks must leave the pretrained function unchanged at init."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from simdist.modeling import lora as lora_mod
from simdist.utils import model as model_utils
from tests import helpers


def _outputs(model, x):
    y = model(x, deterministic=True)
    return {k: np.asarray(v) for k, v in y.items()}


@pytest.mark.parametrize("cfg", helpers.WORLD_MODEL_CONFIGS, indirect=True)
def test_lora_zero_init_is_identity(cfg):
    model = helpers.make_dummy_model(cfg)
    x = helpers.make_dummy_world_model_input(cfg, 3)
    before = _outputs(model, x)
    n_before = model_utils.count_params(model, nnx.Param)
    n = model.add_lora({"rank": 1, "scale": 1.0, "target": "dynamics"}, nnx.Rngs(1))
    assert n == 2 * (8 + 2)  # 2 decoder layers x (2 attentions x 4 projections + 2 MLP linears)
    after = _outputs(model, x)
    for k in before:
        np.testing.assert_allclose(after[k], before[k], rtol=0, atol=0)
    # adapter parameters: rank x (64+64) per attention projection (8 per layer), (64+256)+(256+64) per MLP
    n_lora = lora_mod.lora_param_count(model.dynamics)
    assert n_lora == 2 * (8 * 128 + 320 + 320)
    assert model_utils.count_params(model, nnx.Param) == n_before + n_lora
    filt = model_utils.create_param_filter(["lora_a", "lora_b"])
    assert model_utils.count_params(model, filt) == n_lora
    # a nonzero B changes the dynamics output but not the policy head
    for layer in model.dynamics.layers:
        lin = layer.dense_proj.output_layer
        lin.lora_b.value = jnp.ones_like(lin.lora_b.value)
    changed = _outputs(model, x)
    assert not np.allclose(changed["latents"], before["latents"])
    np.testing.assert_allclose(changed["actions"], before["actions"])


@pytest.mark.parametrize("cfg", helpers.WORLD_MODEL_CONFIGS, indirect=True)
def test_lora_split_merge_roundtrip(cfg):
    model = helpers.make_dummy_model(cfg)
    model.add_lora({"rank": 4, "scale": 1.0, "target": "dynamics"}, nnx.Rngs(2))
    for layer in model.dynamics.layers:
        layer.attention_1.query.lora_b.value = 0.1 * jnp.ones_like(layer.attention_1.query.lora_b.value)
    x = helpers.make_dummy_world_model_input(cfg, 2)
    ref = _outputs(model, x)
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()
    model2 = nnx.merge(graphdef, state)
    np.testing.assert_allclose(_outputs(model2, x)["latents"], ref["latents"])
    # a freshly built model with the same lora config has the same parameter structure (checkpoint restore path)
    cfg2 = dict(cfg)
    cfg2["model"] = dict(model.model_cfg)
    model3 = helpers.make_dummy_model(cfg2)
    _, state3 = nnx.split(model3)
    keys = lambda s: sorted(str(p) for p, _ in jax.tree_util.tree_flatten_with_path(s.filter(nnx.Param).to_pure_dict())[0])
    assert keys(state3) == keys(state)


@pytest.mark.parametrize("cfg", helpers.WORLD_MODEL_CONFIGS, indirect=True)
def test_param_masks(cfg):
    model = helpers.make_dummy_model(cfg)
    full = model_utils.count_params(model, model_utils.create_param_filter(["dynamics"]))
    affine = model_utils.count_params(model, model_utils.create_param_filter(["dynamics/layernorm_1", "dynamics/layernorm_2", "dynamics/layernorm_3"]))
    last = model_utils.count_params(model, model_utils.create_param_filter(["dynamics/1/dense_proj"]))
    assert full == 133504 and affine == 2 * 3 * 2 * 64 and last == 64 * 256 + 256 + 256 * 64 + 64
