"""Design doc 34 (docs/design34_condition_screen_ko.md §4): pick one level per condition family from the screen.

    python scripts/sim2sim/stage34_select.py --d results/stage34 --conds "legF50 legF30 ..."

Per run (diag_<cond>_<arm>.npz): mid-walk = steps with >= 300 steps since the reset.
  walks(level)  = policy-alone mid-walk falls per 1000 steps <= 0.5  and  mid-walk share of the time >= 0.5
  chosen level  = among the walking levels the one with the lowest policy-alone mid-walk reward (most degraded);
                  if no level walks, the level with the fewest policy-alone falls per 1000 steps
"""
import argparse, os
import numpy as np

FAMILY = {"legF70": "leg", "legF50": "leg", "legR50": "leg", "delay1": "delay", "delay2": "delay", "slope3": "slope", "slope5": "slope",
          "tow10": "tow", "tow20": "tow", "mot50": "motor"}
WALK_HAZ = 0.5; WALK_SHARE = 0.5; MID = 300


def metrics(d, name):
    f = os.path.join(d, f"diag_{name}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    s = z["since"].astype(np.int64); fl = z["fall"].astype(bool); ok = ~z["resetting"].astype(bool); mid = ok & (s >= MID)
    T, N = s.shape; n_mid = mid.sum(0).astype(float); rew = z["reward"].astype(float); vx = z["lin_vel"][..., 0].astype(float); vy = z["lin_vel"][..., 1].astype(float)
    sp = z["speeds"].astype(float)
    return {"T": T, "falls": 1000.0 * fl.sum(0) / T, "f_mid": (fl & (s >= MID)).sum(0).astype(float), "n_mid": n_mid,
            "r_mid": (rew * mid).sum(0), "vx_mid": (vx * mid).sum(0), "vy_mid": (np.abs(vy) * mid).sum(0), "speed": sp,
            "start_falls": 1000.0 * (fl & (s < MID)).sum(0) / T}


def agg(x, i=slice(None)):
    """seed-pooled summary over the robots in i"""
    n = max(1.0, x["n_mid"][i].sum())
    return {"falls": float(x["falls"][i].mean()), "start": float(x["start_falls"][i].mean()), "haz": float(1000.0 * x["f_mid"][i].sum() / n),
            "rew": float(x["r_mid"][i].sum() / n), "share": float(x["n_mid"][i].sum() / (x["T"] * len(x["n_mid"][i]))),
            "vratio": float(np.mean(x["vx_mid"][i] / np.maximum(1.0, x["n_mid"][i]) / x["speed"][i])), "vy": float(x["vy_mid"][i].sum() / n)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="results/stage34"); ap.add_argument("--conds", default=" ".join(FAMILY))
    a = ap.parse_args(); fam = {}
    for c in a.conds.split():
        x = metrics(a.d, f"{c}_POL")
        if x is None:
            print(f"family {FAMILY[c]}: {c} has no policy-alone run"); continue
        g = agg(x); g["walks"] = g["haz"] <= WALK_HAZ and g["share"] >= WALK_SHARE
        fam.setdefault(FAMILY[c], []).append((c, g))
        print(f"family {FAMILY[c]}: {c} policy alone mid-walk falls {g['haz']:.2f}, share {g['share']:.2f}, reward {g['rew']:.4f}, walks {g['walks']}")
    with open(os.path.join(a.d, "chosen.txt"), "w") as fh:
        for k, lv in fam.items():
            w = [t for t in lv if t[1]["walks"]]
            c = min(w, key=lambda t: t[1]["rew"])[0] if w else min(lv, key=lambda t: t[1]["falls"])[0]
            fh.write(f"{k} {c} {'walks' if w else 'none_walks'}\n"); print(f"chosen {k}: {c} ({'walks' if w else 'no level walks'})")


if __name__ == "__main__":
    main()
