#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def collect_layer_list(js):
    if isinstance(js, list):  # rare
        return js
    if not isinstance(js, dict):
        return []
    for k in ("layer_diagnostics","diagnostics","layers"):
        if k in js and isinstance(js[k], list):
            return js[k]
    # last resort: any list value of dict-like layers
    for v in js.values():
        if isinstance(v, list) and v and (isinstance(v[0], dict) or hasattr(v[0], "__dict__")):
            return v
    return []

def as_dict(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    # attribute fallbacks
    out = {}
    for k in ("energy","smoothness_index","spectral_entropy","hfer","fiedler_value","fiedler"):
        if hasattr(x, k):
            out[k] = getattr(x, k)
    return out

def summarize_fiedler(f):
    f = np.asarray([v for v in f if v is not None], dtype=float)
    if f.size == 0:
        return dict(fiedler_mean=np.nan, fiedler_peak=np.nan, fiedler_auc=np.nan,
                    fiedler_early=np.nan, fiedler_late=np.nan, fiedler_slope=np.nan)
    L = len(f); k = max(1, L//3)
    return dict(
        fiedler_mean=float(f.mean()),
        fiedler_peak=float(f.max()),
        fiedler_auc=float(f.sum()),
        fiedler_early=float(f[:k].mean()),
        fiedler_late=float(f[-k:].mean()),
        fiedler_slope=float((f[-1]-f[0])/max(1, L-1))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="e.g., analysis/cot")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--perlayer_csv", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rows_perlayer = []
    rows_summary = []

    for model_dir in sorted(out_root.iterdir()):
        if not model_dir.is_dir(): continue
        model = model_dir.name
        for style_dir in sorted(model_dir.iterdir()):
            if not style_dir.is_dir(): continue
            style = style_dir.name  # standard|cot|tot|cod
            files = sorted(style_dir.glob("diagnostics_*.json"))
            if not files: 
                continue
            # use the latest diagnostics (or loop all if you prefer)
            fp = files[-1]
            with open(fp, "r", encoding="utf-8") as f:
                js = json.load(f)
            layers = collect_layer_list(js)
            if not layers:
                continue
            fvals = []
            for li, entry in enumerate(layers):
                d = as_dict(entry)
                fv = d.get("fiedler_value", d.get("fiedler"))
                fvals.append(fv)
                rows_perlayer.append({
                    "model": model, "style": style, "layer": li,
                    "fiedler_value": fv,
                    "energy": d.get("energy"),
                    "smoothness_index": d.get("smoothness_index"),
                    "spectral_entropy": d.get("spectral_entropy"),
                    "hfer": d.get("hfer"),
                    "source": str(fp)
                })
            rows_summary.append({
                "model": model, "style": style, "n_layers": len(layers), "source": str(fp),
                **summarize_fiedler(fvals)
            })

    if not rows_perlayer:
        raise SystemExit("No diagnostics found under the expected structure.")

    dfL = pd.DataFrame(rows_perlayer).sort_values(["model","style","layer"])
    dfS = pd.DataFrame(rows_summary).sort_values(["model","style"])

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.perlayer_csv).parent.mkdir(parents=True, exist_ok=True)

    dfS.to_csv(args.summary_csv, index=False)
    dfL.to_csv(args.perlayer_csv, index=False)

    print(f"Saved summary:   {args.summary_csv}")
    print(f"Saved per-layer: {args.perlayer_csv}")

if __name__ == "__main__":
    main()
