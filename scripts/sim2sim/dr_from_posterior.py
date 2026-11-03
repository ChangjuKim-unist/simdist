#!/usr/bin/env python3
"""Design doc 12: DR range from an identification posterior (budget-allocation rule).

Rule (no hand-set width):
  1. bias correction   p_c(xi) = mean_{b in B} p(xi + b)      B = MAP-true offsets measured in sim (leave-one-out)
  2. budget allocation q(xi) ∝ p_c(xi)^(1/(1+k))               k = width-cost exponent, falls ∝ density^(-k) (fitted)
  3. DR range          = central 90 % interval of q, rounded outward to the posterior grid

Usage: dr_from_posterior.py POSTERIOR_JSON --k K [--bias 0.05 0.10] [--mass 0.9]
Prints "lo,hi" (for GO1_MOTOR_DR) and a JSON summary on stderr.
"""
import argparse
import json
import sys

import numpy as np


def rule(grid, post, k, bias, mass=0.9):
    grid = np.asarray(grid)
    post = np.asarray(post, dtype=float)
    post = post / post.sum()
    step = float(np.round(grid[1] - grid[0], 6))
    # 1. bias correction: p_c(xi) = mean_b p(xi + b)
    pc = np.zeros_like(post)
    for b in bias:
        s = int(round(b / step))
        shifted = np.zeros_like(post)
        if s >= 0:
            shifted[: len(post) - s] = post[s:]
        else:
            shifted[-s:] = post[: len(post) + s]
        pc += shifted / len(bias)
    pc = pc / pc.sum()
    # 2. allocation
    q = pc ** (1.0 / (1.0 + k))
    q = q / q.sum()
    # 3. central interval
    cdf = np.cumsum(q)
    lo_i = int(np.searchsorted(cdf, (1 - mass) / 2, side="right"))
    hi_i = int(np.searchsorted(cdf, 1 - (1 - mass) / 2, side="left"))
    lo_i = max(0, min(lo_i, len(grid) - 1))
    hi_i = max(0, min(hi_i, len(grid) - 1))
    return float(grid[lo_i]), float(grid[hi_i]), pc, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("posterior_json")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--bias", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--axis", type=int, default=0, help="column of candidates holding the identified axis")
    a = ap.parse_args()
    d = json.load(open(a.posterior_json))
    cand = np.asarray(d["candidates"], dtype=float)
    grid = cand[:, a.axis] if cand.ndim == 2 else cand
    grid = np.round(grid, 4)
    lo, hi, pc, q = rule(grid, d["posterior"], a.k, a.bias, a.mass)
    print(f"{lo:.2f},{hi:.2f}")
    json.dump(
        {
            "source": a.posterior_json,
            "k": a.k,
            "bias": a.bias,
            "mass": a.mass,
            "map": d.get("map"),
            "range": [lo, hi],
            "grid": grid.tolist(),
            "q": [round(float(x), 5) for x in q],
        },
        sys.stderr,
    )
    print(file=sys.stderr)


if __name__ == "__main__":
    main()
