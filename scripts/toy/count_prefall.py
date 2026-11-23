"""Design 42: count adaptation data in evalcl *_trans.npz files collected with reset-after-fall.
usable = transitions inside non-overlapping L-step windows that end at or before the first fall (what train --trunc_fall keeps)
robot  = control steps the robot actually ran (each episode stops at its first fall or at T) = real-time budget used.
Prints: usable robot n_episodes n_fell_episodes"""
import sys
import numpy as np

L = 8
usable = robot = n = nf = 0
for p in sys.argv[1:]:
    fl = np.load(p)["fell"].astype(bool)          # (E, T)
    E, T = fl.shape
    upto = np.where(fl.any(1), fl.argmax(1) + 1, T)
    usable += int(np.sum((upto // L) * L))
    robot += int(np.sum(upto))
    n += E
    nf += int(fl.any(1).sum())
print(usable, robot, n, nf)
