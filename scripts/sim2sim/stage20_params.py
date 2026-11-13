"""Design doc 20 fact check: which parameters the fine-tuning filter selects, and the attention-layer internals.

    JAX_PLATFORMS=cpu python scripts/sim2sim/stage20_params.py go1_paper [--heads dynamics] [--verbose]
"""
import argparse, os, flax, jax
import flax.nnx as nnx
from simdist.utils import model as mu, paths

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--heads", default="dynamics"); ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
print("flax", flax.__version__, "jax", jax.__version__)
m, cfg, step = mu.load_model_from_ckpt(os.path.join(paths.get_model_checkpoints_dir(), a.ckpt))
filt = mu.create_param_filter(a.heads.split(",")) if a.heads else nnx.Param
flat = jax.tree_util.tree_flatten_with_path(nnx.state(m).filter(filt).to_pure_dict())[0]
def key(p): return "/".join(str(getattr(x, "key", getattr(x, "idx", x))) for x in p)
tot = sum(v.size for _, v in flat)
if a.verbose:
    for p, v in flat: print(f"{key(p)}: {tuple(v.shape)} {v.size}")
print(f"{a.ckpt} step {step}: selected {tot} params in {len(flat)} tensors; all Param {mu.count_params(m, nnx.Param)}")
print("layernorm affine:", sum(v.size for p, v in flat if "layernorm" in key(p)), "| last layer dense_proj:", sum(v.size for p, v in flat if "layers/1/dense_proj" in key(p)))
print("ckpt cfg: replay", (cfg.get("adapt", {}) or {}).get("replay"), "| loss", (cfg.get("loss", {}) or {}).get("type"), "| heads", cfg.get("heads"), "| warmup", cfg["training"].get("warmup_steps"), "| lr", cfg["training"].get("learning_rate"))
att = m.dynamics.layers[0].attention_1
print(type(att).__name__, [x for x in vars(att) if not x.startswith("_")])
for n in ["query", "key", "value", "out"]:
    sub = getattr(att, n, None)
    print(" ", n, type(sub).__name__, "in", getattr(sub, "in_features", None), "out", getattr(sub, "out_features", None), "axis", getattr(sub, "axis", None), "kernel", tuple(sub.kernel.value.shape), "bias", None if sub.bias is None else tuple(sub.bias.value.shape))
lin = m.dynamics.layers[0].dense_proj.layers[0]
print("  mlp linear", type(lin).__name__, lin.in_features, lin.out_features, tuple(lin.kernel.value.shape))
