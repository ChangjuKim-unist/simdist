"""Motor-switch trace analysis (design doc 14 S0-4): falls before/after the switch and the one-step latent
error around it, per arm.

    python scripts/sim2sim/switch_trace.py /data1/cjkim/stage13/S_*_switch_motor0.4.npz
"""
import sys
import numpy as np

for p in sys.argv[1:]:
    d = np.load(p)
    fall = d["fall"].astype(float)            # (T, N)
    resetting = d["resetting"].astype(bool)
    sw = int(d["motor_switch_step"])
    T, N = fall.shape
    pre = fall[:sw].sum() / (sw * N) * 1000
    post = fall[sw:].sum() / ((T - sw) * N) * 1000
    post_early = fall[sw:sw + 500].sum() / (500 * N) * 1000
    # exact one-step latent error of the executed plan: pred_latents[t, :, 0] vs z[t+1]
    pl = d["pred_latents"].astype(np.float32)  # (T, N, 25, 64)
    z = d["z"].astype(np.float32)              # (T, N, 64)
    err = np.linalg.norm(pl[:-1, :, 0] - z[1:], axis=-1)  # (T-1, N)
    valid = ~resetting[:-1] & ~resetting[1:] & (np.abs(pl[:-1, :, 0]).sum(-1) > 0)
    def m(a, b):
        e, v = err[a:b], valid[a:b]
        return float(e[v].mean()) if v.any() else float("nan")
    name = p.split("/")[-1].replace(".npz", "")
    print(f"{name}: falls/1k pre-switch {pre:.2f}  post-switch(all) {post:.2f}  post(first 500) {post_early:.2f} | "
          f"|e| pre[-500:] {m(sw-500, sw):.3f}  post[0:25] {m(sw, sw+25):.3f}  post[25:100] {m(sw+25, sw+100):.3f}  "
          f"post[100:500] {m(sw+100, sw+500):.3f}  post[500:] {m(sw+500, T):.3f}")
