"""CPU smoke test of the context slot (design doc 13): a pretrained model with a freshly attached
slot must compute exactly the pretrained outputs (zero-initialized injections), the world-model
loss with the context terms must be finite, a gradient step must run, and the oracle override
must change psi[:, :3] only.  Run inside the container with JAX_PLATFORMS=cpu."""
import os, sys, yaml, numpy as np, jax, jax.numpy as jnp
import flax.nnx as nnx
from simdist.utils import model as model_utils, paths
from simdist.modeling import losses
from simdist.data.dataset import WorldModelDatasetBase

ckpt = os.path.join(paths.get_model_checkpoints_dir(), sys.argv[1] if len(sys.argv) > 1 else "go1_paper")
model, model_cfg, step = model_utils.load_model_from_ckpt(ckpt)
print("loaded", ckpt, "step", step, "context:", getattr(model, "context", None) is not None)

item = WorldModelDatasetBase.get_dummy_item(model_cfg)
B = 4
x = {k: jnp.asarray(np.stack([v] * B), jnp.float32) for k, v in item["model_in"].items()}
y = {k: jnp.asarray(np.stack([v] * B), jnp.float32) for k, v in item["labels"].items()}
y["physics"] = jnp.asarray(np.tile(np.array([[0.4, 0.8, 0.0]], np.float32), (B, 1)))
y["physics_valid"] = jnp.ones((B,), jnp.float32)

y0 = model(x, deterministic=True)
ctx_cfg = yaml.safe_load(open(os.path.join(paths.get_config_dir(), "finetune_model.yaml")))["adapt"]["context"]
if getattr(model, "context", None) is None:
    model.add_context(ctx_cfg, nnx.Rngs(0))
y1 = model(x, deterministic=True)
for k in ("latents", "rewards", "values", "actions"):
    d = float(jnp.max(jnp.abs(y0[k] - y1[k])))
    print(f"{k:8s} max |before-after| = {d:.2e}")
    assert d < 1e-5, k
print("context out", y1["context"].shape, np.asarray(y1["context"][0]))

cfg = {"loss": yaml.safe_load(open(os.path.join(paths.get_config_dir(), "loss", "world_model.yaml")))}
loss_fn = losses.get_loss(cfg)
loss, terms = loss_fn(model, x, y, False)
print("loss", float(loss), {k: float(v) for k, v in terms.items()})
assert np.isfinite(float(loss))

optimizer = nnx.Optimizer(model, __import__("optax").adam(1e-4), wrt=nnx.Param)
@nnx.jit
def train_step(model, optimizer, x, y):
    (l, t), g = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(model, x, y, False)
    optimizer.update(g)
    return l
l1 = float(train_step(model, optimizer, x, y)); l2 = float(train_step(model, optimizer, x, y))
print("train steps", l1, l2)
n_ctx = model_utils.count_params(model, model_utils.create_param_filter(["context"]))
print("context encoder params", n_ctx, "total", model_utils.count_params(model, nnx.Param))

xo = dict(x); xo["context_override"] = jnp.asarray(np.tile(np.asarray(model.scale_xi([0.4, 0.8, 0.0]))[None], (B, 1)))
yo = model(xo, deterministic=True)
print("oracle psi[:3]", np.asarray(yo["context"][0, :3]), "expected", np.asarray(model.scale_xi([0.4, 0.8, 0.0])))
assert np.allclose(np.asarray(yo["context"][:, :3]), np.asarray(model.scale_xi([0.4, 0.8, 0.0]))[None], atol=1e-6)
assert np.allclose(np.asarray(yo["context"][:, 3:]), np.asarray(y1["context"][:, 3:]) if False else np.asarray(model(x, deterministic=True)["context"][:, 3:]), atol=1e-6)
print("unscale_xi round trip", np.asarray(model.unscale_xi(model.scale_xi([0.4, 0.8, 0.0]))))
print("OK")
