"""Design doc 20: a LoRA checkpoint reloads with its adapters, the base weights are untouched and the policy head is unchanged.
    JAX_PLATFORMS=cpu python scripts/sim2sim/stage20_check_ckpt.py smoke20_lora1 go1_paper
"""
import os, sys, numpy as np
from simdist.utils import model as mu, paths
from simdist.modeling import lora
from tests import helpers
name, base = sys.argv[1], sys.argv[2]
m, cfg, step = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), name))
print("lora cfg", cfg["model"].get("lora"), "| step", step, "| lora params", lora.lora_param_count(m.dynamics))
print("max |lora_b| per layer:", [float(np.abs(np.asarray(l.attention_1.query.lora_b.value)).max()) for l in m.dynamics.layers])
m0, cfg0, _ = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), base))
x = helpers.make_dummy_world_model_input(cfg0, 2)
y, y0 = m(x, deterministic=True), m0(x, deterministic=True)
print("latents max diff:", float(np.abs(np.asarray(y["latents"] - y0["latents"])).max()), "| actions equal:", bool(np.allclose(np.asarray(y["actions"]), np.asarray(y0["actions"]))))
k, k0 = m.dynamics.layers[0].dense_proj.output_layer.kernel.value, m0.dynamics.layers[0].dense_proj.output_layer.kernel.value
print("base kernel unchanged:", bool(np.array_equal(np.asarray(k), np.asarray(k0))))
