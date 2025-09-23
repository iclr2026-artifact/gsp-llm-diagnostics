#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def pick_json(p: Path):
    if p.is_file():
        return p
    c = sorted(p.glob("diagnostics_*.json")) or sorted(p.glob("*.json"))
    return c[-1] if c else None

def load_fiedler_series(jp: Path):
    with open(jp, "r", encoding="utf-8-sig") as f:
        d = json.load(f)
    rows = d["diagnostics"] if isinstance(d, dict) and "diagnostics" in d else d
    L = np.array([int(r.get("layer", i)) for i, r in enumerate(rows)], dtype=int)
    V = np.array([float(r.get("fiedler_value", np.nan)) for r in rows], dtype=float)
    return L, V

def delta_early(jA: Path, jB: Path, lo: int, hi: int):
    LA, VA = load_fiedler_series(jA)
    LB, VB = load_fiedler_series(jB)
    common = np.intersect1d(LA, LB)
    m = (common >= lo) & (common <= hi)
    if not m.any(): 
        return np.nan
    ia = {int(l): i for i, l in enumerate(LA)}
    ib = {int(l): i for i, l in enumerate(LB)}
    diffs = [VB[ib[int(l)]] - VA[ia[int(l)]] for l in common[m]]
    return float(np.mean(diffs))

def main():
    ap = argparse.ArgumentParser(description="Build causal baseline-vs-ablate Δλ2(early) summary CSV")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="Triplets: family baseline_dir_or_json ablate_dir_or_json (repeat per family)")
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    if len(args.pairs) % 3 != 0:
        raise SystemExit("Provide triplets: <family> <baseline> <ablate> ...")

    rows=[]
    for i in range(0, len(args.pairs), 3):
        family = args.pairs[i]
        base_p = Path(args.pairs[i+1])
        ablt_p = Path(args.pairs[i+2])
        jb = pick_json(base_p)
        ja = pick_json(ablt_p)
        if jb is None or ja is None:
            print(f"!! Skipping {family}: could not find JSON in {base_p} or {ablt_p}")
            continue
        delt = delta_early(jb, ja, args.early_lo, args.early_hi)
        rows.append({
            "family": family,
            "baseline_json": str(jb),
            "ablate_json": str(ja),
            "early_lo": args.early_lo,
            "early_hi": args.early_hi,
            "delta_lambda2_early": delt  # ablate - baseline
        })

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅ wrote {out}  (n={len(rows)})")

if __name__ == "__main__":
    main()
