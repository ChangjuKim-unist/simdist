"""Add a per-step "terminated" flag to raw sim-to-sim episode logs recorded before the
logger wrote one. An episode ended by a fall unless (a) it is the last episode of its
5000-step collection run (the next file starts more than `gap` seconds of wall-clock
time after it ends, or there is no next file) or (b) it reached the episode time limit.

    python scripts/sim2sim/mark_terminated.py sim2sim_combo_hard_go1_paper [--time_limit 4900] [--gap 20]
"""
import argparse
import glob
import os
from datetime import datetime

import h5py
import numpy as np

from simdist.utils import paths

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--time_limit", type=int, default=4900)
ap.add_argument("--gap", type=float, default=20.0)
ap.add_argument("--rate", type=float, default=50.0)
args = ap.parse_args()

files = glob.glob(os.path.join(paths.get_real_raw_data_dir(args.dataset), "*", "*.hdf5"))
info = []
for f in files:
    with h5py.File(f, "r") as h:
        n = len(h["reward"])
        t = h["time"][:]  # wall-clock nanoseconds written by the logger
        t0, t1 = float(t[0]) / 1e9, float(t[-1]) / 1e9
    info.append((f, t0, n, t1))
info.sort(key=lambda x: x[1])  # by logged start time (file mtimes are not reliable after copies)
counts = {"fall": 0, "run_end": 0, "time_limit": 0}
for i, (f, t0, n, end) in enumerate(info):
    last_of_run = i == len(info) - 1 or info[i + 1][1] - end > args.gap
    if n >= args.time_limit:
        kind = "time_limit"
    elif last_of_run:
        kind = "run_end"
    else:
        kind = "fall"
    counts[kind] += 1
    with h5py.File(f, "a") as h:
        if "terminated" in h:
            del h["terminated"]
        h.create_dataset("terminated", data=np.full((n,), float(kind == "fall"), dtype=np.float32))
print(args.dataset, counts)
