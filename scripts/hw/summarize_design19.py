"""Design 19 (docs/design19_real_failure_ko.md): trial-sheet summary, branch verdict and randomized arm order.

The primary metric (forward distance, m) is read by a person from floor marks, so the input is a hand-filled
CSV trial sheet with columns

    arm,surface,slope_deg,speed,trial,distance_m,success,fell,note

(arm in P1..P5; surface in slip, foam; success/fell 0 or 1; distance clipped to the surface target).
Optional: --logs adds odometry distance from datasets/real/hw_<arm>_<surface><slope>_<speed>/raw/*.hdf5 as a check.

    python scripts/hw/summarize_design19.py results/real/stage19/trials.csv --out results/real/stage19/summary.md
    python scripts/hw/summarize_design19.py --order results/real/stage19/order.csv     # randomized arm order per cell (seed 19)
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

TARGET = {"slip": 1.8, "foam": 3.0}
SPEEDS = {"slip": [0.1, 0.3, 0.5], "foam": [0.2, 0.5, 0.8]}
ARMS = ["P1", "P2", "P3", "P4", "P5"]
ARM_NAMES = {"P1": "논문 설정 DR policy", "P2": "넓은 DR policy", "P3": "SimDist planner zero-shot", "P4": "SimDist 적응", "P5": "표면 DR policy"}
TRIALS = 10
MIN_TRIALS = 5
# section 4 boundaries (fixed before the runs)
FAIL_SUCC, FAIL_DIST = 0.5, 0.6
OK_SUCC, OK_DIST = 0.8, 0.9
GAIN = 0.3


def boot_diff(a, b, n=2000, seed=0):
    """CI of mean(a) - mean(b) with independent resampling of each side (trials are not paired)."""
    rng = np.random.default_rng(seed); a = np.asarray(a, float); b = np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if not r.get("arm"): continue
        rows.append(dict(arm=r["arm"].strip(), surface=r["surface"].strip(), slope=float(r.get("slope_deg") or 0), speed=round(float(r["speed"]), 2),
                         dist=min(float(r["distance_m"]), TARGET[r["surface"].strip()]), succ=int(r["success"]), fell=int(r.get("fell") or 0)))
    return rows


def cells(rows):
    c = defaultdict(list)
    for r in rows: c[(r["arm"], r["surface"], r["slope"], r["speed"])].append(r)
    return c


def final_slope(rows, surface):
    """Surface A is judged at the final (highest) slope that has P2 data; foam at 0."""
    s = [r["slope"] for r in rows if r["surface"] == surface and r["arm"] == "P2"]
    return max(s) if s else 0.0


def arm_stats(c, arm, surface, slope):
    """Per-speed success rate / mean distance ratio; None if any speed has < MIN_TRIALS."""
    out = {}
    for sp in SPEEDS[surface]:
        rs = c.get((arm, surface, slope, sp), [])
        if len(rs) < MIN_TRIALS: return None
        out[sp] = (np.mean([r["succ"] for r in rs]), np.mean([r["dist"] for r in rs]) / TARGET[surface], len(rs))
    return out


def visibly_fails(st):
    return st is not None and (sum(v[0] <= FAIL_SUCC for v in st.values()) >= 2 or np.mean([v[1] for v in st.values()]) <= FAIL_DIST)


def no_problem(st):
    return st is not None and all(v[0] >= OK_SUCC for v in st.values()) and np.mean([v[1] for v in st.values()]) >= OK_DIST


def pooled(c, arm, surface, slope, key):
    return [r[key] for sp in SPEEDS[surface] for r in c.get((arm, surface, slope, sp), [])]


def gain_ok(c, a, b, surface, slope):
    """success(a) - success(b) >= GAIN with CI excluding 0 (pooled over speeds)."""
    xa, xb = pooled(c, a, surface, slope, "succ"), pooled(c, b, surface, slope, "succ")
    if len(xa) < MIN_TRIALS or len(xb) < MIN_TRIALS: return None, None
    m, lo, hi = boot_diff(xa, xb)
    return (m >= GAIN and lo > 0), (m, lo, hi)


def table(rows, c, out):
    out.append("| 팔 | " + " | ".join(f"{s} {sp} m/s" for s in ("slip", "foam") for sp in SPEEDS[s]) + " |")
    out.append("|---|" + "---|" * 6)
    out.append("칸 = 전진 거리 평균 m (목표 slip 1.8 / foam 3.0, 클수록 좋음) / 성공 횟수 / 시행 수. slip은 최종 경사의 칸.")
    for arm in ARMS:
        cols = []
        for s in ("slip", "foam"):
            sl = final_slope(rows, s) if s == "slip" else 0.0
            for sp in SPEEDS[s]:
                rs = c.get((arm, s, sl, sp), [])
                cols.append(f"{np.mean([r['dist'] for r in rs]):.2f} / {sum(r['succ'] for r in rs)} / {len(rs)}" if rs else "—")
        out.append(f"| {arm} {ARM_NAMES[arm]} | " + " | ".join(cols) + " |")


def verdict(rows, c, out):
    out.append("\n**§4 판정** (경계: 실패 = 세 속도 중 둘 이상 성공률 ≤ 0.5 또는 평균 거리 ≤ 목표의 0.6; 문제 없음 = 모든 속도 성공률 ≥ 0.8 그리고 거리 ≥ 0.9; 이득 = 성공률 차 ≥ +0.3, CI 0 제외)")
    branch = {}
    for s in ("slip", "foam"):
        sl = final_slope(rows, s) if s == "slip" else 0.0
        st = {a: arm_stats(c, a, s, sl) for a in ARMS}
        p2f, p2ok = visibly_fails(st["P2"]), no_problem(st["P2"])
        g42, ci42 = gain_ok(c, "P4", "P2", s, sl); g45, ci45 = gain_ok(c, "P4", "P5", s, sl)
        p4ok = st["P4"] is not None and np.mean([v[1] for v in st["P4"].values()]) >= OK_DIST
        line = f"- {s} (경사 {sl:g}°): P2 {'실패' if p2f else ('문제 없음' if p2ok else '중간/자료 부족')}"
        if ci42: line += f"; P4 − P2 성공률 {ci42[0]:+.2f} [{ci42[1]:+.2f}, {ci42[2]:+.2f}]"
        if ci45: line += f"; P4 − P5 {ci45[0]:+.2f} [{ci45[1]:+.2f}, {ci45[2]:+.2f}]"
        if p2ok: b = "갈래 1 후보 (P2 문제 없음)"
        elif p2f and g42 and p4ok and (visibly_fails(st["P5"]) or g45): b = "갈래 2 (P2 실패, P4가 고침, P5가 못 고치거나 P4보다 뚜렷이 못함)"
        elif p2f and g42 and p4ok and st["P5"] is not None and not g45: b = "갈래 1에 합침 (P4가 고치지만 표면 DR P5도 비슷 → 표면을 DR에 넣는 게 더 쌈)"
        elif p2f and visibly_fails(st["P4"]) and visibly_fails(st["P5"]): b = "갈래 3 (P2·P4·P5 모두 실패)"
        else: b = "불명 (칸당 10회 추가 후 재판정)"
        branch[s] = b; out.append(line + f" → {b}")
    if all("갈래 1 후보" in b for b in branch.values()): out.append("- **종합: 갈래 1 (분석 논문)** — 두 표면 모두 넓은 DR policy에 문제 없음")
    elif any(b.startswith("갈래 2") for b in branch.values()): out.append("- **종합: 갈래 2 (방법 논문)** — 한 표면에서 성립")
    elif any(b.startswith("갈래 3") for b in branch.values()): out.append("- **종합: 갈래 3 (범위 밖)**")
    else: out.append("- **종합: 불명 또는 갈래 1에 합침** — §4 표 참조")


def order(path, seed=19):
    rng = np.random.default_rng(seed)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["day", "surface", "speed", "trial", "arm_order"])
        for day, arms in ((1, ["P1", "P2"]), (1, ["P3"]), (2, ["P4", "P5"])):
            for s in ("slip", "foam"):
                for sp in SPEEDS[s]:
                    for t in range(1, TRIALS + 1):
                        w.writerow([day, s, sp, t, " ".join(rng.permutation(arms))])
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sheet", nargs="?"); ap.add_argument("--out"); ap.add_argument("--order")
    a = ap.parse_args()
    if a.order: order(a.order); return
    if not a.sheet: ap.error("sheet or --order")
    rows = load(a.sheet); c = cells(rows); out = [f"# 설계 19 결과 ({os.path.basename(a.sheet)}, 시행 {len(rows)})\n"]
    table(rows, c, out); verdict(rows, c, out)
    falls = {arm: (sum(r["fell"] for r in rows if r["arm"] == arm), sum(1 for r in rows if r["arm"] == arm)) for arm in ARMS}
    out.append("\n넘어짐 (전체 시행 중, 작을수록 좋음): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in falls.items() if v[1]))
    text = "\n".join(out); print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); open(a.out, "w").write(text + "\n"); print("wrote", a.out)


if __name__ == "__main__":
    main()
