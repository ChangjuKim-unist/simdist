# Design doc 25 check 1: steps since the last reset at every fall, per speed, from planner trace npz files.
# usage: python3 scripts/sim2sim/fall_timing.py <trace.npz> ...
import sys, numpy as np, collections
for f in sys.argv[1:]:
    d = np.load(f, allow_pickle=True)
    fall = d['fall']; reset = d['reset']; resetting = d['resetting']; speeds = d['speeds']
    T, N = fall.shape
    model = str(d['model']); cond = str(d['condition'])
    per = collections.defaultdict(list)
    for j in range(N):
        last = 0
        for t in range(T):
            if fall[t, j]:
                per[float(speeds[j])].append(t - last)
            if t > 0 and reset[t, j]:
                last = t
    print(f"== {f.split('/')[-1]}  model={model} cond={cond}  T={T} N={N}")
    for sp in sorted(per):
        a = np.array(per[sp])
        if len(a) == 0: continue
        print(f"  speed {sp:.1f}: falls {len(a):4d} | since-reset median {np.median(a):6.0f} | <=100: {np.mean(a<=100):.2f} | <=200: {np.mean(a<=200):.2f} | >500: {np.mean(a>500):.2f}")
