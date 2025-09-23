#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# --- Import the real adapter (works whether invoked from repo root or not)
try:
    from scripts.run_model_adapter import run_model_once  # preferred
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _P
    here = _P(__file__).resolve().parent
    repo = here.parent
    if str(here) not in sys.path: sys.path.insert(0, str(here))
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    from run_model_adapter import run_model_once  # fallback


def delta_lambda2(diag_a, diag_b, lo=2, hi=5):
    """
    Mean (B - A) Fiedler value over layers [lo, hi].
    diag_* must be dicts with key "diagnostics": [{"layer": int, "fiedler_value": float, ...}, ...]
    """
    rows_a = diag_a["diagnostics"]
    rows_b = diag_b["diagnostics"]
    La = np.array([int(r.get("layer", i)) for i, r in enumerate(rows_a)])
    Lb = np.array([int(r.get("layer", i)) for i, r in enumerate(rows_b)])
    Va = np.array([float(r["fiedler_value"]) for r in rows_a], float)
    Vb = np.array([float(r["fiedler_value"]) for r in rows_b], float)
    common = np.intersect1d(La, Lb)
    m = (common >= lo) & (common <= hi)
    if not m.any():
        return np.nan
    ia = {int(l): i for i, l in enumerate(La)}
    ib = {int(l): i for i, l in enumerate(Lb)}
    diffs = [Vb[ib[int(l)]] - Va[ia[int(l)]] for l in common[m]]
    return float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser(
        description="Causal intervention: flip voice cues, control tokenization, measure Δλ2 and behavior."
    )
    ap.add_argument("--pairs", required=True, help="JSONL from flip_voice_minimal.py")
    ap.add_argument("--model", required=True, help="model id (e.g., llama-3.2-1b)")
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    # BOM-tolerant read because PowerShell often writes UTF-8 with BOM
    with open(args.pairs, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("drift_ok", False):
                continue  # skip pairs where tokenization drift exceeded threshold

            lang = r["lang"]
            # A = original; B = flipped
            diag_A, beh_A = run_model_once(args.model, r["orig"], lang)
            diag_B, beh_B = run_model_once(args.model, r["flipped"], lang)

            dlam = delta_lambda2(diag_A, diag_B, args.early_lo, args.early_hi)
            beh_delta = (beh_B.get("score", np.nan) - beh_A.get("score", np.nan))

            rows.append({
                "id": r["id"],
                "lang": lang,
                "model": args.model,
                "early_lo": args.early_lo,
                "early_hi": args.early_hi,
                "delta_lambda2_early": dlam,
                "behavior_A": beh_A.get("score", np.nan),
                "behavior_B": beh_B.get("score", np.nan),
                "behavior_delta": beh_delta,
            })

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"✅ wrote {outp}  (n={len(rows)})")


if __name__ == "__main__":
    main()
