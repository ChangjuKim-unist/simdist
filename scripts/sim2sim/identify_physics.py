"""Design doc 10: policy-independent physics identification by replaying logged transitions in the simulator.

For short segments of a real-format log (the deployed system's own states and actions, whichever policy
collected them), every simulated robot is reset to the logged state, the logged actions are applied
open-loop for K steps, and the predicted proprioception is compared with the logged one. Each robot runs
with a different candidate motor-strength scale; the per-candidate error over many segments gives a
likelihood over the scale (DROPO-style, Gaussian with a scale eps). No policy, world model or history
features are involved, so the estimate does not depend on who collected the data.

    python scripts/sim2sim/identify_physics.py dataset=sim2sim_very_weak_motor_priorfirst_b45000 out=results/stage5/id_pf.json --headless
Hydra config: config/identify_physics.yaml
"""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import json, os, time
import h5py, hydra, numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
from simdist.rl.envs import get_sim_env_cfg
from simdist.utils import config, paths

# Go1 leg geometry (go1.urdf): hip joint at (+-0.1881, +-0.04675, 0), hip->thigh offset +-0.08 in y, thigh 0.213, calf 0.213, foot radius 0.01
HIP = np.array([[0.1881, -0.04675, 0], [0.1881, 0.04675, 0], [-0.1881, -0.04675, 0], [-0.1881, 0.04675, 0]]); SIDE = np.array([-1, 1, -1, 1])
L_HIP, L_TH, L_CA, R_FOOT = 0.08, 0.213, 0.213, 0.01
JOINT_ORDER = ["FR", "FL", "RR", "RL"]  # Isaac joint order is FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ... (per-type grouping); resolved at runtime

def foot_positions_base(q):  # q: (N, 4 legs, 3) hip, thigh, calf angles -> (N, 4, 3) foot in base frame
    hx, th, ca = q[..., 0], q[..., 1], q[..., 2]
    # leg plane: thigh/calf rotate about y; hip rotates about x
    x = -L_TH * np.sin(th) - L_CA * np.sin(th + ca)
    z_leg = -L_TH * np.cos(th) - L_CA * np.cos(th + ca)
    y_leg = SIDE * L_HIP
    y = y_leg * np.cos(hx) - z_leg * np.sin(hx); z = y_leg * np.sin(hx) + z_leg * np.cos(hx)
    return HIP[None] + np.stack([x, y, z], -1)

def quat_from_gravity(g):  # projected gravity in base frame (N,3) -> (w,x,y,z) with yaw 0 (ZYX: pitch = asin(gx), roll = atan2(-gy, -gz))
    gx, gy, gz = (g / np.linalg.norm(g, axis=-1, keepdims=True)).T
    pitch = np.arcsin(np.clip(gx, -1, 1)); roll = np.arctan2(-gy, -gz)
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.stack([cr * cp, sr * cp, cr * sp, -sr * sp], -1), roll, pitch

def rotmat(roll, pitch):
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Rx = np.stack([np.ones_like(cr), 0 * cr, 0 * cr, 0 * cr, cr, -sr, 0 * cr, sr, cr], -1).reshape(-1, 3, 3)
    Ry = np.stack([cp, 0 * cp, sp, 0 * cp, np.ones_like(cp), 0 * cp, -sp, 0 * cp, cp], -1).reshape(-1, 3, 3)
    return Ry @ Rx

@hydra.main(config_path="../../config", config_name="identify_physics", version_base=None)
def main(cfg):
    d = os.path.join(paths.get_real_dataset_dir(cfg.dataset), f"processed_data_{cfg.system}_H-25_T-25")
    P = h5py.File(os.path.join(d, "proprio_obs.hdf5"))["data"][:]; A = h5py.File(os.path.join(d, "acts.hdf5"))["data"][:]
    starts = h5py.File(os.path.join(d, "start_idxs.hdf5"))["data"][:]; sc = json.load(open(os.path.join(d, "scaler_params.json")))
    std = np.asarray(sc["proprio_obs"]["std"], np.float32) + 1e-6
    grid = np.linspace(cfg.grid.lo, cfg.grid.hi, cfg.grid.n); G, R, K = len(grid), int(cfg.replicas), int(cfg.seg_len); N = G * R
    scale_per_env = np.repeat(grid, R)
    env_cfg = get_sim_env_cfg(cfg.system); env_cfg.scene.num_envs = N; env_cfg.episode_length_s = 1000.0
    tg = env_cfg.scene.terrain.terrain_generator; tg.curriculum = False; tg.size = (15.0, 15.0); tg.num_rows = 1; tg.num_cols = 1; tg.difficulty_range = (0.0, 0.0)
    tg.sub_terrains = {"plane": tg.sub_terrains["plane"]}
    env_cfg.events.physics_material.params["static_friction_range"] = (cfg.friction, cfg.friction); env_cfg.events.physics_material.params["dynamic_friction_range"] = (cfg.friction, cfg.friction)
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (cfg.add_mass, cfg.add_mass)
    env = ManagerBasedRLEnv(env_cfg); env.reset()
    robot = env.scene["robot"]; dev = env.device
    act = robot.actuators["base_legs"]; nominal = act.effort_limit.clone()
    act.effort_limit[:] = nominal * torch.as_tensor(scale_per_env, device=dev, dtype=nominal.dtype)[:, None]
    names = ["base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "cos_sin_phase"]  # go1 proprio layout (35 dims)
    jn = robot.joint_names; default_q = robot.data.default_joint_pos[0].cpu().numpy()
    leg_idx = [[jn.index(f"{l}_{p}_joint") for p in ("hip", "thigh", "calf")] for l in JOINT_ORDER]  # (4,3) indices into the 12 joints
    origins = env.scene.env_origins.cpu().numpy()
    rng = np.random.default_rng(int(cfg.seed)); err_sum = np.zeros(N); err_cnt = np.zeros(N); t0 = time.time()
    # segments: steps s..s+K inside a window (so inside one episode)
    for rnd in range(int(cfg.rounds)):
        s = rng.choice(starts, N) + rng.integers(0, 25 - K, N) if K < 25 else rng.choice(starts, N)
        p0 = P[s]  # (N,35)
        q = default_q[None] + p0[:, 9:21]; dq = p0[:, 21:33]
        quat, roll, pitch = quat_from_gravity(p0[:, 6:9]); Rm = rotmat(roll, pitch)
        feet = foot_positions_base(q[:, np.array(leg_idx)])  # (N,4,3) using Isaac joint index map per leg
        feet_w = np.einsum("nij,nkj->nki", Rm, feet); z = -(feet_w[..., 2].min(1)) + R_FOOT
        pos = origins + np.stack([np.zeros(N), np.zeros(N), z], -1)
        v_w = np.einsum("nij,nj->ni", Rm, p0[:, 0:3]); w_w = np.einsum("nij,nj->ni", Rm, p0[:, 3:6])
        ids = torch.arange(N, device=dev)
        robot.write_root_pose_to_sim(torch.as_tensor(np.concatenate([pos, quat], -1), device=dev, dtype=torch.float32), ids)
        robot.write_root_velocity_to_sim(torch.as_tensor(np.concatenate([v_w, w_w], -1), device=dev, dtype=torch.float32), ids)
        robot.write_joint_state_to_sim(torch.as_tensor(q, device=dev, dtype=torch.float32), torch.as_tensor(dq, device=dev, dtype=torch.float32), env_ids=ids)
        env.scene.write_data_to_sim()
        seg_err = np.zeros(N); alive = np.ones(N, bool)
        for k in range(K):
            obs, _, term, trunc = env.step(torch.as_tensor(A[s + k], device=dev, dtype=torch.float32))[0:4]
            alive &= ~(term | trunc).cpu().numpy()
            pred = np.concatenate([obs["obs"][n].cpu().numpy() for n in names], -1)
            e = ((pred - P[s + k + 1]) / std)[:, :33]  # exclude the phase clock
            if k >= int(cfg.skip): seg_err += np.mean(e ** 2, -1)
        err_sum += np.where(alive, seg_err / max(K - int(cfg.skip), 1), 0); err_cnt += alive
        if rnd % 5 == 0: print(f"round {rnd}: alive {alive.mean():.2f}, mean err {np.nanmean(err_sum[err_cnt > 0] / err_cnt[err_cnt > 0]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    per_env = err_sum / np.maximum(err_cnt, 1); per_cand = per_env.reshape(G, R).mean(1); cnt = err_cnt.reshape(G, R).sum(1)
    eps = np.median(per_cand); post = np.exp(-(per_cand - per_cand.min()) / (2 * (0.1 * eps) ** 2)); post /= post.sum()
    res = {"dataset": cfg.dataset, "grid": grid.tolist(), "error": per_cand.tolist(), "segments": cnt.tolist(), "posterior": post.tolist(),
           "map": float(grid[int(np.argmin(per_cand))]), "mean": float((grid * post).sum()), "seg_len": K, "rounds": int(cfg.rounds)}
    print("error per candidate:", " ".join(f"{g:.2f}:{e:.3f}" for g, e in zip(grid, per_cand)))
    print(f"MAP motor scale {res['map']:.2f}, posterior mean {res['mean']:.2f}")
    if cfg.out: os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True); json.dump(res, open(cfg.out, "w"), indent=2)
    env.close()

if __name__ == "__main__":
    main(); app.close()
