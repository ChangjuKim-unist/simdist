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
    grid_m = np.linspace(cfg.grid.lo, cfg.grid.hi, cfg.grid.n)
    grid_f = np.array(list(cfg.friction_grid) if cfg.get("friction_grid") else [cfg.friction], np.float64)
    grid_k = np.array(list(cfg.mass_grid) if cfg.get("mass_grid") else [cfg.add_mass], np.float64)
    cand = np.array([(m, f, k) for m in grid_m for f in grid_f for k in grid_k]); G, R, K = len(cand), int(cfg.replicas), int(cfg.seg_len); N = G * R
    cand_env = np.repeat(cand, R, axis=0); scale_per_env = cand_env[:, 0]
    env_cfg = get_sim_env_cfg(cfg.system); env_cfg.scene.num_envs = N; env_cfg.episode_length_s = 1000.0
    tg = env_cfg.scene.terrain.terrain_generator; tg.curriculum = False; tg.size = (15.0, 15.0); tg.num_rows = 1; tg.num_cols = 1; tg.difficulty_range = (0.0, 0.0)
    tg.sub_terrains = {"plane": tg.sub_terrains["plane"]}
    env_cfg.events.physics_material.params["static_friction_range"] = (cfg.friction, cfg.friction); env_cfg.events.physics_material.params["dynamic_friction_range"] = (cfg.friction, cfg.friction)
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (cfg.add_mass, cfg.add_mass)
    env_cfg.seed = int(cfg.seed)
    for ev in ("joint_stiffness_and_damping", "joint_friction"):  # nominal Kp/Kd/joint friction for every candidate (no per-reset draws)
        if hasattr(env_cfg.events, ev): setattr(env_cfg.events, ev, None)
    env = ManagerBasedRLEnv(env_cfg); env.reset()
    robot = env.scene["robot"]; dev = env.device
    act = robot.actuators["base_legs"]; nominal = act.effort_limit.clone()
    act.effort_limit[:] = nominal * torch.as_tensor(scale_per_env, device=dev, dtype=nominal.dtype)[:, None]
    if len(grid_f) > 1:  # per-env foot friction through the material event's shape bookkeeping
        ev = [t for t in env.event_manager._mode_term_cfgs["startup"] if hasattr(t.func, "num_shapes_per_body")][0].func
        mats = robot.root_physx_view.get_material_properties(); fr = torch.as_tensor(cand_env[:, 1], dtype=mats.dtype)
        for body_id in ev.asset_cfg.body_ids:
            a = sum(ev.num_shapes_per_body[:body_id]); b = a + ev.num_shapes_per_body[body_id]
            mats[:, a:b, 0] = fr[:, None]; mats[:, a:b, 1] = fr[:, None]
        robot.root_physx_view.set_material_properties(mats, torch.arange(N))
    if len(grid_k) > 1:  # per-env added base mass
        masses = robot.root_physx_view.get_masses(); base_id = robot.find_bodies("trunk")[0][0] if "trunk" in robot.body_names else robot.find_bodies("base")[0][0]
        masses[:, base_id] = robot.data.default_mass[:, base_id].cpu() + torch.as_tensor(cand_env[:, 2], dtype=masses.dtype)
        robot.root_physx_view.set_masses(masses, torch.arange(N))
    names = ["base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "cos_sin_phase"]  # go1 proprio layout (35 dims)
    jn = robot.joint_names; default_q = robot.data.default_joint_pos[0].cpu().numpy()   # articulation (Isaac) joint order
    # the log's joint_pos / joint_vel / actions are in Unitree order (simdist.rl.go2.JOINT_NAMES, preserve_order=True);
    # u2n[i] = articulation index of Unitree joint i. Before 2026-11-03 the two orders were added together (bug).
    from simdist.rl.go2 import JOINT_NAMES
    u2n = np.array([jn.index(n) for n in JOINT_NAMES]); default_u = default_q[u2n]
    print("articulation joint order:", jn); print("Unitree -> articulation index map:", u2n.tolist())
    leg_idx = [[jn.index(f"{l}_{p}_joint") for p in ("hip", "thigh", "calf")] for l in JOINT_ORDER]  # (4,3) articulation indices
    origins = env.scene.env_origins.cpu().numpy()
    rng = np.random.default_rng(int(cfg.seed)); t0 = time.time(); RND = int(cfg.rounds)
    E = np.zeros((RND, N)); W = np.zeros((RND, N)); EB = np.zeros((RND, N)); NB = np.zeros((RND, N)); fell = np.zeros((RND, N), bool)
    padded = ~np.any(A != 0, axis=-1)  # all-zero action rows = the aggregator's frozen 25-step prefix, not real motion
    def sample_starts():
        off = lambda n: rng.integers(0, 25 - K, n) if K < 25 else 0
        s = rng.choice(starts, N) + off(N)
        for _ in range(20):  # resample windows that overlap the padded prefix
            bad = padded[s] | padded[s + K - 1]
            if not bad.any(): break
            s[bad] = rng.choice(starts, int(bad.sum())) + off(int(bad.sum()))
        return s
    # segments: steps s..s+K inside a window (so inside one episode)
    for rnd in range(RND):
        s = sample_starts()
        p0 = P[s]  # (N,35)
        q = np.repeat(default_q[None], N, 0).astype(np.float32); dq = np.zeros((N, 12), np.float32)
        q[:, u2n] = default_u[None] + p0[:, 9:21]; dq[:, u2n] = p0[:, 21:33]   # Unitree-ordered log -> articulation order
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
        seg_err = np.zeros(N); seg_err_b = np.zeros(N); seg_w = np.zeros(N); alive = np.ones(N, bool)
        for k in range(K):
            a_k = A[s + k]
            obs, _, term, trunc = env.step(torch.as_tensor(a_k, device=dev, dtype=torch.float32))[0:4]
            alive &= ~(term | trunc).cpu().numpy()   # a candidate that falls keeps accumulating error (its post-reset obs vs the log)
            pred = np.concatenate([obs["obs"][n].cpu().numpy() for n in names], -1)
            tgt = P[s + k + 1]
            # torque demand of the logged PD command (Kp (q_des - q) - Kd dq), Unitree order; the motor cap only matters where it
            # binds, so steps are weighted by how close the demand comes to the nominal limit
            q_des = default_u[None] + float(cfg.action_scale) * a_k; q_log = default_u[None] + P[s + k][:, 9:21]
            tau = float(cfg.kp) * (q_des - q_log) - float(cfg.kd) * P[s + k][:, 21:33]
            w_k = np.clip(np.abs(tau).max(-1) / float(cfg.tau_nominal), 0.0, 1.0) ** 2 if cfg.demand_weight else np.ones(N)
            e = (pred - tgt) / std
            if k >= int(cfg.skip):
                seg_err += w_k * np.mean(e[:, 9:33] ** 2, -1); seg_w += w_k           # joint channel (motor): demand-weighted
                seg_err_b += np.mean(e[:, 0:6] ** 2, -1)                               # base-velocity channel (friction / mass): unweighted
        E[rnd] = seg_err; W[rnd] = seg_w; EB[rnd] = seg_err_b; NB[rnd] = K - int(cfg.skip); fell[rnd] = ~alive
        if rnd % 5 == 0:
            cur = (E[:rnd + 1].sum(0) / np.maximum(W[:rnd + 1].sum(0), 1e-8)).reshape(G, R).mean(1)
            print(f"round {rnd}: alive {alive.mean():.2f}, best {cand[int(np.argmin(cur))].round(2).tolist()} err {cur.min():.3f} ({time.time()-t0:.0f}s)", flush=True)
    # demand-weighted error over ALL segments of a candidate (sum w*e / sum w, replicas pooled); low-demand segments count little.
    # (Before 2026-11-03 each segment was normalized by its own weight sum, so a no-demand segment counted as much as a saturated one.)
    def cand_err(rows):
        e = E[rows].sum(0).reshape(G, R).sum(1); w = W[rows].sum(0).reshape(G, R).sum(1)
        eb = EB[rows].sum(0).reshape(G, R).sum(1); nb = NB[rows].sum(0).reshape(G, R).sum(1)
        return e / np.maximum(w, 1e-8), eb / np.maximum(nb, 1e-8)
    per_cand, per_cand_b = cand_err(np.arange(RND)); cnt = (~fell).reshape(RND, G, R).sum((0, 2))
    score = per_cand + (per_cand_b if cfg.base_channel else 0.0); best = cand[int(np.argmin(score))]
    # posterior := bootstrap distribution of the argmin over resampled rounds. No likelihood temperature: the old
    # Gaussian with eps = 0.1 * median(error over the grid) was grid-dependent and over-confident.
    # Two data-calibrated posteriors (no temperature constant):
    #  (a) bootstrap_argmin: how often each candidate is the best over resampled rounds (discrete, zero tails);
    #  (b) posterior (used for the DR-width rule): exp(-z^2/2) with z = (mean excess error over the best) / (its bootstrap
    #      standard error), i.e. how many standard errors a candidate is worse than the best. Continuous tails.
    B = int(cfg.get("bootstrap", 200)); D = np.zeros((B, G))
    for b in range(B):
        rows = rng.integers(0, RND, RND); e_b, eb_b = cand_err(rows); tot = e_b + (eb_b if cfg.base_channel else 0.0)
        D[b] = tot - tot.min()
    post_arg = np.bincount(np.argmin(D, 1), minlength=G) / B
    z = D.mean(0) / np.maximum(D.std(0), 1e-8); z[np.argmin(score)] = 0.0
    post = np.exp(-0.5 * z ** 2); post /= post.sum()
    mean = (cand * post[:, None]).sum(0); per_cand = score
    res = {"dataset": cfg.dataset, "candidates": cand.tolist(), "error": score.tolist(), "segments": cnt.tolist(), "posterior": post.tolist(),
           "map": best.tolist(), "mean": mean.tolist(), "seg_len": K, "rounds": RND, "axes": ["motor", "friction", "mass"],
           "posterior_type": "z_score_vs_best", "bootstrap": B, "bootstrap_argmin": post_arg.tolist(), "z": z.tolist(),
           "joint_order_map": u2n.tolist()}
    if len(grid_f) == 1 and len(grid_k) == 1:
        print("error per candidate:", " ".join(f"{g:.2f}:{e:.3f}" for g, e in zip(grid_m, per_cand)))
    else:  # marginal posteriors per axis
        for j, (ax, g) in enumerate(zip(["motor", "friction", "mass"], [grid_m, grid_f, grid_k])):
            marg = np.array([post[cand[:, j] == v].sum() for v in g]); print(f"{ax:9s} marginal: " + " ".join(f"{v:.2f}:{m_:.2f}" for v, m_ in zip(g, marg)))
    print(f"MAP (motor, friction, mass) = {best.round(2).tolist()}, posterior mean = {mean.round(2).tolist()}")
    if cfg.out: os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True); json.dump(res, open(cfg.out, "w"), indent=2)
    env.close()

if __name__ == "__main__":
    main(); app.close()
