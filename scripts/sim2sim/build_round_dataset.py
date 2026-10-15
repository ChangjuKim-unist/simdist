"""Build the raw directory of an Algorithm-1 round-k dataset from earlier rounds.

    python scripts/sim2sim/build_round_dataset.py <out_name> <round1_dataset>:<max_steps> <round2_dataset>[:<max_steps>] ...

Creates datasets/real/<out_name>/raw/<i>_<source>/ with relative symlinks to the
first <max_steps> steps (in collection order) of each source dataset's raw
episodes; the episode aggregator then sees them in the same order.
"""
import glob
import os
import sys

import h5py

from simdist.utils import paths

out = sys.argv[1]
root = paths.get_real_data_dir() if hasattr(paths, "get_real_data_dir") else os.path.dirname(paths.get_real_dataset_dir("x"))
raw_out = os.path.join(root, out, "raw")
for i, spec in enumerate(sys.argv[2:], start=1):
    src, _, budget = spec.partition(":")
    budget = int(budget) if budget else None
    files = sorted(glob.glob(os.path.join(root, src, "raw", "*", "*.hdf5")), key=os.path.getmtime)
    sub = os.path.join(raw_out, f"{i}_{src}")
    os.makedirs(sub, exist_ok=True)
    total = 0
    n = 0
    for f in files:
        if budget is not None and total >= budget:
            break
        with h5py.File(f, "r") as h:
            total += len(h["reward"])
        dst = os.path.join(sub, os.path.basename(f))
        if not os.path.lexists(dst):
            os.symlink(os.path.relpath(f, sub), dst)
        n += 1
    print(f"{src}: {n} episodes, {total} steps -> {sub}")
