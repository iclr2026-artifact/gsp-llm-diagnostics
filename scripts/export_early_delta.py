#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob

METRIC_KEY = "fiedler_value"

def pick_json(pathlike: str) -> str | None:
    p = Path(pathlike)
    cands = []
    if any(ch in pathlike for ch in "*?[]"):
        cands = [Path(x) for x in glob(pathlike)]
    elif p.is_dir():
        cands = sorted(p.glob("diagnostics_*.json")) or sorted(p.glob("*.json"))
    elif p.is_file():
        cands = [p]
    else:
        return None
    if not cands:
        return None
    diags = [c for c in cands if c.name.startswith("diagnostics_")]
    srcs = diags if diags else cands
    return str(sorted(srcs, key=lambda x: x.stat().st_mtime)[-1])

def extract_layers_metric(jpath: str, metric_key: str):
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "diagnostics" in data:
        L, V = [], []
        for d in data["diagnostics"]:
            if isinstance(d, dict) and metric_key in d:
                L.append(int(d.get("layer", len(L))))
                V.append(float(d[metric_key]))
        return np.array(L), np.array(V, float)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and "layers" in v:
                L = [int(x.get("layer", i)) for i, x in enumerate(v["layers"])]
                V = [float(x.get(metric_key, np.nan)) for x in v["layers"]]
                return np.array(L), np.array(V, float)
    if isinstance(data, list) and data and isinstance(data[0], dict) and metric_key in data[0]:
        L = [int(x.get("layer", i)) for i, x in enumerate(data)]
        V = [float(x.get(metric_key, np.nan)) for x in data]
        return np.array(L), np.array(V, float)
    raise ValueError(f"Unrecognized JSON or missing metric '{metric_key}': {jpath}")

def align_delta(active_path: str, passive_path: str, metric_key: str):
    ap = pick_json(active_path); pp = pick_json(passive_path)
    if ap is None or pp is None:
        raise FileNotFoundError(f"Missing JSON: active={active_path}, passive={passive_path}")
    La, Va = extract_layers_metric(ap, metric_key)
    Lp, Vp = extract_layers_metric(pp, metric_key)
    common = np.intersect1d(La, Lp)
    if common.size == 0:
        raise ValueError(f"No shared layers: {ap} vs {pp}")
    ia = {int(l): i for i,l in enumerate(La)}
    ip = {int(l): i for i,l in enumerate(Lp)}
    Va_c = np.array([Va[ia[int(l)]] for l in common], float)
    Vp_c = np.array([Vp[ip[int(l)]] for l in common], float)
    return common.astype(int), (Vp_c - Va_c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True, help="Folder with *_per_language.csv manifests")
    ap.add_argument("--models", nargs="+", required=True, help="Model keys used in manifest filenames, e.g. qwen2.5-7b")
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_rows = []
    root = Path(args.manifests)
    for mk in args.models:
        mf = root / f"{mk}_per_language.csv"
        if not mf.exists():
            print(f"-- skip (missing manifest): {mf}")
            continue
        df = pd.read_csv(mf)
        # tolerate both schemas
        act_col = "active_json" if "active_json" in df.columns else ("active" if "active" in df.columns else None)
        pas_col = "passive_json" if "passive_json" in df.columns else ("passive" if "passive" in df.columns else None)
        if not act_col or not pas_col:
            print(f"-- skip (no active/passive columns) in {mf}")
            continue
        for _, r in df.iterrows():
            label = str(r["label"]).upper()
            a = str(r[act_col]); p = str(r[pas_col])
            try:
                L, d = align_delta(a, p, METRIC_KEY)
            except Exception as e:
                print(f"!! {mk}/{label}: {e}")
                continue
            mask = (L >= args.early_lo) & (L <= args.early_hi)
            if not mask.any():
                continue
            early_mean = float(np.nanmean(d[mask]))
            out_rows.append(dict(model=mk, label=label, early_lo=args.early_lo,
                                 early_hi=args.early_hi, delta_lambda2_mean=early_mean))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f" wrote {out}  ({len(out_rows)} rows)")

if __name__ == "__main__":
    main()
