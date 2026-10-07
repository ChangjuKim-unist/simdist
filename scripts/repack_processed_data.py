"""Rewrite the hdf5 files of a processed dataset with a contiguous layout.

Processed datasets written before the time-only chunking in
``simdist.data.data_processor`` use HDF5's automatic chunks, which split the
feature dimension. Every random window read then pulls in far more data than
needed, which makes training disk-bound once the dataset is larger than RAM.
This script rewrites each file with a contiguous (unchunked) layout so a
window read touches only the rows it needs. Files are replaced in place, one
at a time, so free disk space of at least the largest file is required.

Usage:
    python scripts/repack_processed_data.py <processed_data_dir> [--rows-per-block N]
"""

import argparse
import glob
import os

import h5py
import numpy as np
from tqdm import tqdm


def repack_file(path: str, rows_per_block: int) -> None:
    tmp_path = path + ".repack.tmp"
    with h5py.File(path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for key in src:
            s = src[key]
            if s.chunks is None:
                print(f"{os.path.basename(path)}:{key} is already contiguous, skipping")
                continue
            d = dst.create_dataset(key, shape=s.shape, dtype=s.dtype)  # contiguous
            n = s.shape[0]
            desc = f"{os.path.basename(path)} ({s.nbytes / 1e9:.1f} GB)"
            for i in tqdm(range(0, n, rows_per_block), desc=desc):
                j = min(i + rows_per_block, n)
                d[i:j] = s[i:j]
    with h5py.File(tmp_path, "r") as dst, h5py.File(path, "r") as src:
        for key in src:
            assert dst[key].shape == src[key].shape, key
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("processed_data_dir")
    parser.add_argument("--rows-per-block", type=int, default=200_000)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.processed_data_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No hdf5 files in {args.processed_data_dir}")
    # smallest first so a failure on the big file leaves the rest done
    files.sort(key=os.path.getsize)
    for path in files:
        if os.path.exists(path + ".repack.tmp"):
            os.remove(path + ".repack.tmp")
        repack_file(path, args.rows_per_block)
    print("Done!")


if __name__ == "__main__":
    main()
