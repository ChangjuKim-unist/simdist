"""Design 40 toy: evaluate the blind DR PPO policy under a condition (same seeds/metrics as toy_mppi).

  PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs python scripts/toy/toy_ppo_eval.py \
      --params /mnt/newssd/cjkim/toy/runs/ppo_s0/params.pkl --cond nominal --out results/stage40/e0_ppo_nominal
"""
import argparse, json, os, pickle, time

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params

import toy_common as C


def load_policy(env, path):
    with open(path, "rb") as f:
        params = pickle.load(f)
    p = locomotion_params.brax_ppo_config(C.ENV)
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **p.network_factory)
    return ppo_networks.make_inference_fn(nets)(params, deterministic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--cond", default="nominal", choices=C.CONDITIONS)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    env = C.load_env()
    F = C.Feat(env)
    env._mjx_model = C.condition_model(env, a.cond)      # true condition dynamics
    pi = load_policy(env, a.params)

    def set_cmd(st, t):
        info = dict(st.info)
        info["command"] = jnp.array([C.v_des(t), 0.0, 0.0])
        info["steps_until_next_cmd"] = jnp.array(10 ** 6, jnp.int32)
        return st.replace(info=info)

    def episode(seed):
        st = env.reset(jax.random.PRNGKey(seed))
        st = set_cmd(st, 0.0)
        key = jax.random.PRNGKey(2000 + seed)

        def body(carry, step):
            st, key = carry
            key, k = jax.random.split(key)
            act, _ = pi(st.obs, k)
            st = env.step(st, act)
            st = set_cmd(st, (step + 1) * env.dt)
            rec = C.step_log(F, st.data)
            rec["ctrl"] = F.default_q + act * env._config.action_scale   # joint targets, comparable to MPPI
            return (st, key), rec

        _, recs = jax.lax.scan(body, (st, key), jnp.arange(a.T))
        return recs

    t = time.time()
    recs = jax.jit(jax.vmap(episode))(jnp.arange(a.seed0, a.seed0 + a.episodes))
    recs = jax.tree_util.tree_map(np.asarray, recs)
    wall = time.time() - t
    summ = C.summarise(recs, recs["ctrl"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out + ".npz", **recs)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summ.items()}
    js.update(cond=a.cond, arm="ppo", wall_s=wall, args=vars(a))
    with open(a.out + ".json", "w") as f:
        json.dump(js, f, indent=1)
    print(f"ppo {a.cond}: J={np.mean(summ['J']):.3f} fell={np.mean(summ['fell']):.2f} "
          f"vx={np.nanmean(summ['vx_mean']):.3f} duty={np.nanmean(summ['duty'], 0).round(2)} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
