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


def early_delta(diag_active, diag_passive, lo=2, hi=5):
    """Mean (passive - active) λ2 over the early window [lo, hi]."""
    rows_a = diag_active["diagnostics"]; rows_p = diag_passive["diagnostics"]
    La = np.array([int(r.get("layer", i)) for i, r in enumerate(rows_a)])
    Lp = np.array([int(r.get("layer", i)) for i, r in enumerate(rows_p)])
    Va = np.array([float(r["fiedler_value"]) for r in rows_a], float)
    Vp = np.array([float(r["fiedler_value"]) for r in rows_p], float)
    common = np.intersect1d(La, Lp)
    m = (common >= lo) & (common <= hi)
    if not m.any():
        return np.nan
    ia = {int(l): i for i, l in enumerate(La)}
    ip = {int(l): i for i, l in enumerate(Lp)}
    return float(np.mean([Vp[ip[int(l)]] - Va[ia[int(l)]] for l in common[m]]))

def early_var(diag, lo=2, hi=5):
    """Variance of λ2 within the early window (stability proxy). Lower = more stable."""
    rows = diag["diagnostics"]
    L = np.array([int(r.get("layer", i)) for i, r in enumerate(rows)])
    V = np.array([float(r["fiedler_value"]) for r in rows], float)
    m = (L >= lo) & (L <= hi)
    if not m.any():
        return np.nan
    return float(np.var(V[m]))


def main():
    ap = argparse.ArgumentParser(
        description="Δλ2-guided selection: choose between active/passive per item and record scores."
    )
    ap.add_argument("--dataset", required=True,
                    help="JSONL with fields: id, lang, active, passive, gold (gold optional for this script)")
    ap.add_argument("--models", nargs="+", required=True,
                    help="e.g., llama-3.2-1b qwen2.5-7b")
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    ap.add_argument("--select_rule", choices=["sign", "var"], default="sign",
                    help=(
                        "How to pick the prompt:\n"
                        "  sign: choose 'passive' if Δλ2(passive-active) > 0 else 'active'.\n"
                        "  var:  compute λ2 variance per variant in [lo,hi]; pick lower-variance (more stable)."
                    ))
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # BOM-tolerant read (PowerShell sometimes writes UTF-8 with BOM)
    with open(args.dataset, "r", encoding="utf-8-sig") as f:
        items = [json.loads(x) for x in f if x.strip()]

    results = []
    for model in args.models:
        for rec in items:
            lang = rec["lang"]
            a = rec["active"]
            p = rec["passive"]
            # Run once per variant; reuse diagnostics and behavior
            diag_a, beh_a = run_model_once(model, a, lang)
            diag_p, beh_p = run_model_once(model, p, lang)

            d = early_delta(diag_a, diag_p, args.early_lo, args.early_hi)

            if args.select_rule == "sign":
                # If Δ=Vp-Va > 0, passive has higher λ2 (more connected); choose passive, else active.
                chosen = "passive" if (not np.isnan(d) and d > 0) else "active"
            else:  # "var"
                var_a = early_var(diag_a, args.early_lo, args.early_hi)
                var_p = early_var(diag_p, args.early_lo, args.early_hi)
                # pick lower variance; if tie/NaN, fall back to sign rule
                if np.isfinite(var_a) and np.isfinite(var_p) and var_a != var_p:
                    chosen = "active" if var_a < var_p else "passive"
                else:
                    chosen = "passive" if (not np.isnan(d) and d > 0) else "active"

            score_chosen = beh_p.get("score", np.nan) if chosen == "passive" else beh_a.get("score", np.nan)

            results.append({
                "id": rec.get("id", ""),
                "lang": lang,
                "model": model,
                "early_lo": args.early_lo,
                "early_hi": args.early_hi,
                "select_rule": args.select_rule,
                "delta_lambda2_early": d,
                "var_active": early_var(diag_a, args.early_lo, args.early_hi),
                "var_passive": early_var(diag_p, args.early_lo, args.early_hi),
                "score_active": beh_a.get("score", np.nan),
                "score_passive": beh_p.get("score", np.nan),
                "score_chosen": score_chosen,
                "chosen": chosen,
            })

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(outp, index=False)
    print(f"✅ wrote {outp}  (n={len(results)})")


if __name__ == "__main__":
    main()
