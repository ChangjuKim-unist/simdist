import os, numpy as np, jax, jax.numpy as jnp
from omegaconf import OmegaConf
from simdist.utils import model as mu, paths
from simdist.control.mppi_batch import BatchedMppiController
ck=lambda n: os.path.join(paths.get_model_checkpoints_dir(), n)
m_mot, cfg, _ = mu.load_model_from_ckpt(ck("go1_paper_very_weak_motor_dyn_b45000_s42"))
m_mot2, _, _ = mu.load_model_from_ckpt(ck("go1_paper_very_weak_motor_dyn_b45000_s42"))
m_pre, _, _ = mu.load_model_from_ckpt(ck("go1_paper"))
ctrl = OmegaConf.to_container(OmegaConf.load("config/control/mppi.yaml"))
N=3
def make(extra, gamma):
    c = BatchedMppiController(m_mot, cfg, ctrl, N, [11,12,13])
    rng=np.random.default_rng(0)
    c.reset(range(N), rng.normal(size=(N,c.proprio_dim)).astype(np.float32)*0.1, np.zeros((N,c.extero_dim),np.float32), np.zeros((N,c.act_dim),np.float32), np.tile([[0.5,0,0]],(N,1)).astype(np.float32))
    c.robust_models=list(extra); c.robust_gamma=np.asarray(gamma,np.float32)
    return np.asarray(c.run_control())
a0=make([], [0,0,0]); a1=make([m_mot2],[1,1,1]); a2=make([m_pre],[0,0,0]); a3=make([m_pre],[1,1,1]); a4=make([m_pre],[1,0,0.5])
print("same model gamma1 == none:", np.abs(a1-a0).max())
print("other model gamma0 == none:", np.abs(a2-a0).max())
print("other model gamma1 differs:", np.abs(a3-a0).max(axis=(1,2)))
print("per-robot gamma [1,0,.5]: robot1 equals none:", np.abs(a4[1]-a0[1]).max(), "robot0 equals gamma1:", np.abs(a4[0]-a3[0]).max())
# numerical check: same weights in two module instances, inside one jit
c = BatchedMppiController(m_mot, cfg, ctrl, N, [11,12,13])
rng=np.random.default_rng(0)
c.reset(range(N), rng.normal(size=(N,c.proprio_dim)).astype(np.float32)*0.1, np.zeros((N,c.extero_dim),np.float32), np.zeros((N,c.act_dim),np.float32), np.tile([[0.5,0,0]],(N,1)).astype(np.float32))
x=c._model_inputs(); x=dict(x); x["fut_acts"]=jnp.asarray(rng.normal(size=(N,c.T,c.act_dim)).astype(np.float32))
r1=np.asarray(jax.jit(lambda xx: m_mot.inference(xx)["rewards"])(x)); r2=np.asarray(jax.jit(lambda xx: m_mot2.inference(xx)["rewards"])(x))
print("reward diff same weights:", np.abs(r1-r2).max(), "scale", np.abs(r1).max())
b0=make([], [0,0,0]); b1=make([], [0,0,0])
print("repeat none vs none:", np.abs(b0-b1).max(), np.abs(b0-a0).max())
def both(xx):
    xr = jax.tree.map(lambda a: jnp.repeat(a, 472, axis=0), xx)
    xr["fut_acts"] = xr["fut_acts"] + 0.3*jax.random.normal(jax.random.key(0), xr["fut_acts"].shape)
    ya = m_mot.inference(xr); yb = m_mot2.inference(xr)
    ra = jnp.sum(ya["rewards"]*c.discounts,-1)+ya["values"][:,-1]*c.final_discount
    rb = jnp.sum(yb["rewards"]*c.discounts,-1)+yb["values"][:,-1]*c.final_discount
    return jnp.abs(ra-rb).max(), jnp.abs(ra).max(), jnp.sort(ra)[-64:]
d, sc, top = jax.jit(both)(x); top=np.asarray(top)
print("in-jit return diff:", float(d), "scale", float(sc), "smallest gap among top-64:", np.diff(np.sort(top)).min())
