#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

def _pick_json(pathlike) -> str:
    """Return the newest diagnostics JSON path from a file/dir/glob."""
    p = Path(pathlike)
    candidates = []
    s = str(pathlike)

    if any(ch in s for ch in "*?[]"):
        candidates = [Path(x) for x in glob(s)]
    elif p.is_dir():
        candidates = sorted(p.glob("diagnostics_*.json")) or sorted(p.glob("*.json"))
    elif p.is_file():
        candidates = [p]

    if not candidates:
        raise FileNotFoundError(f"No JSON found at {pathlike}")

    # Prefer diagnostics_*; pick newest by mtime
    diags = [c for c in candidates if c.name.startswith("diagnostics_")]
    if diags:
        return str(sorted(diags, key=lambda x: x.stat().st_mtime)[-1])
    return str(sorted(candidates, key=lambda x: x.stat().st_mtime)[-1])

def load_series(pathlike, metric_key: str = "fiedler_value"):
    jp = _pick_json(pathlike)
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case A: {"diagnostics": [ {layer:..., <metric>:...}, ... ]}
    if isinstance(data, dict) and isinstance(data.get("diagnostics"), list):
        rows = [r for r in data["diagnostics"] if isinstance(r, dict)]
        L = np.array([int(r.get("layer", i)) for i, r in enumerate(rows)], int)
        V = np.array([float(r.get(metric_key, np.nan)) for r in rows], float)
        return L, V

    # Case B: {"layers":[{layer:..., <metric>:...}, ...]} possibly nested under a key
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and isinstance(v.get("layers"), list):
                rows = [r for r in v["layers"] if isinstance(r, dict)]
                L = np.array([int(r.get("layer", i)) for i, r in enumerate(rows)], int)
                V = np.array([float(r.get(metric_key, np.nan)) for r in rows], float)
                return L, V

    # Case C: top-level list of dicts
    if isinstance(data, list) and data and isinstance(data[0], dict):
        rows = [r for r in data if isinstance(r, dict)]
        L = np.array([int(r.get("layer", i)) for i, r in enumerate(rows)], int)
        V = np.array([float(r.get(metric_key, np.nan)) for r in rows], float)
        return L, V

    raise ValueError(f"Unrecognized JSON schema for {jp} (no '{metric_key}').")

def window_mean(layers: np.ndarray, diffs: np.ndarray, lo: int, hi: int) -> float:
    mask = (layers >= lo) & (layers <= hi)
    return float(np.nanmean(diffs[mask])) if mask.any() else np.nan

def main():
    ap = argparse.ArgumentParser(description="Numeric Δλ2 summaries (ablate - baseline)")
    ap.add_argument("--baseline", required=True, help="dir with diagnostics_*.json OR a JSON file")
    ap.add_argument("--ablate",   required=True, help="dir with diagnostics_*.json OR a JSON file")
    ap.add_argument("--out_csv",  required=True, help="per-layer table (CSV)")
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    args = ap.parse_args()

    jb = _pick_json(args.baseline)
    ja = _pick_json(args.ablate)

    Lb, Vb = load_series(jb, metric_key="fiedler_value")
    La, Va = load_series(ja, metric_key="fiedler_value")

    common = np.intersect1d(Lb, La)
    if common.size == 0:
        raise SystemExit("No overlapping layers between baseline and ablate JSONs.")

    ib = {int(l): i for i, l in enumerate(Lb)}
    ia = {int(l): i for i, l in enumerate(La)}

    rows = []
    for L in common:
        base = float(Vb[ib[int(L)]])
        abl  = float(Va[ia[int(L)]])
        rows.append({
            "layer": int(L),
            "lambda2_base": base,
            "lambda2_ablate": abl,
            "delta": abl - base,  # ablate - base
        })

    df = pd.DataFrame(rows).sort_values("layer")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Ranges
    early_lo = args.early_lo
    early_hi = args.early_hi
    mid_lo   = early_hi + 1
    mid_hi   = early_hi + 5
    late_lo  = early_hi + 6
    late_hi  = int(df["layer"].max())

    Early   = window_mean(df["layer"].values, df["delta"].values, early_lo, early_hi)
    Mid     = window_mean(df["layer"].values, df["delta"].values, mid_lo, mid_hi)
    Late    = window_mean(df["layer"].values, df["delta"].values, late_lo, late_hi)
    Overall = float(np.nanmean(df["delta"].values))

    print(f"✅ wrote {args.out_csv}  (layers={len(df)})")
    print(f"Δλ2 early [{early_lo},{early_hi}] = {Early:.6f}")
    print(f"Δλ2 mid   [{mid_lo},{mid_hi}] = {Mid:.6f}")
    print(f"Δλ2 late  [{late_lo},{late_hi}] = {Late:.6f}")
    print(f"Δλ2 overall = {Overall:.6f}")

if __name__ == "__main__":
    main()
