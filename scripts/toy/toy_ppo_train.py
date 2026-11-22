"""Design 40 toy: train the blind DR PPO baseline (MuJoCo Playground Go1 joystick).

Uses Playground's own Go1JoystickFlatTerrain PPO config and domain randomizer unchanged.
Run on gpu1 inside the container:
  PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs python scripts/toy/toy_ppo_train.py --out /mnt/newssd/cjkim/toy/runs/ppo_s0 --seed 0
"""
import argparse, functools, json, os, pickle, time

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

ENV = "Go1JoystickFlatTerrain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0 = Playground default")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    env = registry.load(ENV)
    env_cfg = registry.get_default_config(ENV)
    p = locomotion_params.brax_ppo_config(ENV)
    if a.steps:
        p.num_timesteps = a.steps
    kw = dict(p)
    net = ppo_networks.make_ppo_networks
    if "network_factory" in kw:
        del kw["network_factory"]
        net = functools.partial(ppo_networks.make_ppo_networks, **p.network_factory)
    kw["seed"] = a.seed
    log, t0 = [], time.time()

    def progress(step, m):
        r = float(m.get("eval/episode_reward", float("nan")))
        log.append({"step": int(step), "reward": r, "t": time.time() - t0})
        print(f"step {step:>11d}  eval reward {r:8.2f}  {time.time() - t0:7.0f}s", flush=True)

    train = functools.partial(ppo.train, **kw, network_factory=net,
                              randomization_fn=registry.get_domain_randomizer(ENV),
                              progress_fn=progress)
    _, params, _ = train(environment=env, eval_env=registry.load(ENV, config=env_cfg),
                         wrap_env_fn=wrapper.wrap_for_brax_training)
    with open(os.path.join(a.out, "params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(params), f)
    cfg = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in p.items()}
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump({"env": ENV, "seed": a.seed, "ppo": cfg, "log": log}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
