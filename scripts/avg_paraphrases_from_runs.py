#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np

def pick_diag_json(run_dir: Path):
    cands = sorted(run_dir.glob("diagnostics_*.json"))
    if not cands:
        cands = sorted(run_dir.glob("*.json"))
    return cands[-1] if cands else None

def load_series(json_path: Path, metric_keys):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # format 1: {"diagnostics":[{"layer":...,"fiedler_value":...}, ...]}
    if isinstance(data, dict) and "diagnostics" in data and isinstance(data["diagnostics"], list):
        rows = data["diagnostics"]
    # format 2: {"model": {"layers":[{...}, ...]}}
    elif isinstance(data, dict) and len(data)==1 and isinstance(list(data.values())[0], dict) and "layers" in list(data.values())[0]:
        rows = list(data.values())[0]["layers"]
    # format 3: just a list of layer dicts
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        rows = data
    else:
        raise ValueError(f"Unrecognized JSON: {json_path}")

    layers = [int(r.get("layer", i)) for i, r in enumerate(rows)]
    out = { "layer": np.array(layers, dtype=int) }
    for k in metric_keys:
        out[k] = np.array([float(r.get(k, np.nan)) for r in rows], dtype=float)
    return out

def average_over_runs(run_dirs, metric_keys):
    """Align by common layers across runs; return dict with averaged metrics."""
    series = []
    for rd in run_dirs:
        jp = pick_diag_json(rd)
        if not jp: 
            continue
        series.append(load_series(jp, metric_keys))
    if not series:
        return None

    # intersect layers
    common = series[0]["layer"]
    for s in series[1:]:
        common = np.intersect1d(common, s["layer"])
    if common.size == 0:
        return None

    # stack per metric
    avg = {"diagnostics": []}
    for L in common:
        row = {"layer": int(L)}
        for k in metric_keys:
            vals = []
            for s in series:
                idx = np.where(s["layer"] == L)[0]
                if idx.size:
                    v = s[k][idx[0]]
                    if np.isfinite(v):
                        vals.append(v)
            row[k] = float(np.mean(vals)) if vals else float("nan")
        avg["diagnostics"].append(row)
    return avg

def main():
    ap = argparse.ArgumentParser(description="Average diagnostics across paraphrase runs into avg_active/avg_passive")
    ap.add_argument("--root_in", required=True, help=r"e.g. .\iclr_results\multi20_expanded")
    ap.add_argument("--root_out", required=True, help=r"e.g. .\iclr_results\multi20_avg")
    ap.add_argument("--metric_keys", nargs="+", default=[
        "fiedler_value","hfer","energy","smoothness_index","spectral_entropy"
    ])
    args = ap.parse_args()

    root_in  = Path(args.root_in)
    root_out = Path(args.root_out)
    root_out.mkdir(parents=True, exist_ok=True)

    # Expect layout: model\lang_mode\sym-*\agg-*\run_*
    for mdl_dir in root_in.iterdir():
        if not mdl_dir.is_dir(): 
            continue
        for lang_mode in mdl_dir.iterdir():
            if not lang_mode.is_dir(): 
                continue
            # split lang_mode: e.g. "en_active"
            m = re.match(r"^([A-Za-z\-]+)_(active|passive)$", lang_mode.name)
            if not m: 
                continue
            lang, mode = m.group(1).lower(), m.group(2)

            # collect run_* under any sym-*/agg-* combos (you can change this to pick only one config)
            run_dirs = []
            for sym_dir in lang_mode.glob("sym-*"):
                for agg_dir in sym_dir.glob("agg-*"):
                    run_dirs.extend([d for d in agg_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

            if not run_dirs:
                print(f"-- no runs: {mdl_dir.name}/{lang}_{mode}")
                continue

            avg = average_over_runs(run_dirs, args.metric_keys)
            if avg is None:
                print(f"-- skip (no common layers): {mdl_dir.name}/{lang}_{mode}")
                continue

            # write to model\lang\avg_active|avg_passive\diagnostics_avg.json
            out_dir = root_out / mdl_dir.name / lang / f"avg_{mode}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "diagnostics_avg.json", "w", encoding="utf-8") as f:
                json.dump(avg, f, ensure_ascii=False, indent=2)
            print(f"✅ wrote {out_dir / 'diagnostics_avg.json'}  (runs={len(run_dirs)})")

if __name__ == "__main__":
    import numpy as np
    main()
