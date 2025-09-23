#!/usr/bin/env python3
import json, argparse, numpy as np
from pathlib import Path

def load_series(p):
    p = Path(p)
    with open(p, "r", encoding="utf-8-sig") as f:
        d = json.load(f)
    rows = d["layer_diagnostics"] if "layer_diagnostics" in d else d.get("diagnostics", d)
    L = np.array([int(r.get("layer", i)) for i, r in enumerate(rows)])
    V = np.array([float(r.get("fiedler_value", r.get("fiedler", "nan"))) for r in rows], float)
    return L, V

def delta_early(a, b, lo, hi):
    La, Va = load_series(a); Lb, Vb = load_series(b)
    common = np.intersect1d(La, Lb)
    m = (common >= lo) & (common <= hi)
    if not m.any(): return float("nan")
    ia = {int(l):i for i,l in enumerate(La)}; ib = {int(l):i for i,l in enumerate(Lb)}
    diffs = [Vb[ib[int(l)]] - Va[ia[int(l)]] for l in common[m]]
    return float(np.mean(diffs))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="diagnostics JSON (baseline)")
    ap.add_argument("--b", required=True, help="diagnostics JSON (ablated/patched)")
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--hi", type=int, default=5)
    args = ap.parse_args()
    print(delta_early(args.a, args.b, args.lo, args.hi))
