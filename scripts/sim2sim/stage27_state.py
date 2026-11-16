# Design doc 27, exploratory (written after the registered results, not used for any verdict): why are planner-made slip states
# more dangerous? State at the slip event, planner-made vs policy-made, and AUC of each feature for the 100-step fall.
#   usage: python3 scripts/sim2sim/stage27_state.py > results/stage27/state_explore.md
import numpy as np, csv, sys
D = 'results/stage27'
def qrot_inv(q, v):  # rotate world vector v into body frame, q = (w,x,y,z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qv = np.stack([x, y, z], 1); t = 2 * np.cross(-qv, v)
    return v + w[:, None] * t + np.cross(-qv, t)
def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool); p, n = x[y], x[~y]
    if len(p) == 0 or len(n) == 0: return np.nan
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
rows = {}
for src in ['AB_P', 'AB_pi', 'AP_P', 'AP_pi']:
    a = np.load(f'{D}/anchor_{src}.npz', allow_pickle=True)
    key = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(a['seed'], a['step']))}
    fl = {}
    for r in csv.DictReader(open(f'{D}/branch_{src}.csv')):
        if r['is_event'] != '1': continue
        k = (int(r['seed']), int(r['step']))
        f100 = r['fell'] == '1' and 0 <= int(r['fall_step']) < 100
        fl.setdefault(k, {}).setdefault(r['group'], []).append(f100)
    idx = np.array([key[k] for k in fl]); q = a['root_pose'][idx, 3:7]
    g = qrot_inv(q, np.tile([0, 0, -1.0], (len(idx), 1)))
    v = qrot_inv(q, a['root_vel'][idx, :3]); w = qrot_inv(q, a['root_vel'][idx, 3:])
    h = a['hist_a'][idx].astype(float)
    feat = dict(slip_ma=a['slip_ma'][idx], tilt=np.linalg.norm(g[:, :2], axis=1), rp_rate=np.linalg.norm(w[:, :2], axis=1),
                yaw_rate=np.abs(w[:, 2]), vx=v[:, 0], abs_vy=np.abs(v[:, 1]), vz=np.abs(v[:, 2]), height=a['root_pose'][idx, 2],
                qd=np.linalg.norm(a['joint_vel'][idx], axis=1),
                hist_jerk=np.linalg.norm(h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2], axis=2).mean(1),
                hist_rate=np.linalg.norm(h[:, 1:] - h[:, :-1], axis=2).mean(1),
                cmd_vx=a['cmd_vel'][idx, 0], since=a['since_reset'][idx])
    out = {gname: np.array([np.mean(fl[k].get(gname, [np.nan])) for k in fl]) for gname in ['A0', 'Fcl', 'Nrep']}
    rows[src] = (feat, out)
names = list(rows['AB_P'][0])
for m, (P, pi) in [('broad', ('AB_P', 'AB_pi')), ('paper', ('AP_P', 'AP_pi'))]:
    print(f'\n## {m}: slip-event state, planner-made vs policy-made (mean, ratio) | AUC of feature for policy-head recovery fall (pooled)')
    fP, oP = rows[P]; fpi, opi = rows[pi]
    yA = np.concatenate([oP['A0'], opi['A0']]) > 0.5; yF = np.concatenate([oP['Fcl'], opi['Fcl']]) > 0.5
    print('| feature | planner-made | policy-made | ratio | AUC policy fall | AUC planner fall |')
    print('|---|---|---|---|---|---|')
    for n in names:
        x = np.concatenate([fP[n], fpi[n]])
        print(f'| {n} | {np.mean(fP[n]):.3f} | {np.mean(fpi[n]):.3f} | {np.mean(fP[n]) / np.mean(fpi[n]):.2f} | {auc(x, yA):.2f} | {auc(x, yF):.2f} |')
    print('falls (n points with clone-mean>0.5): policy', yA.sum(), 'planner', yF.sum(), 'of', len(yA))
