#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def load_layers(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # diagnostics_* shape or plain list
    if isinstance(d, dict) and "diagnostics" in d and isinstance(d["diagnostics"], list):
        layers = d["diagnostics"]
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        layers = d
    else:
        raise ValueError(f"Unrecognized JSON format: {path}")
    return layers

def mean_across_paraphrases(json_paths):
    """Return a list of layer dicts averaged across paraphrases (intersecting layer ids)."""
    all_layers = [load_layers(p) for p in json_paths]
    # layer ids per paraphrase
    layer_sets = [set(int(x.get("layer", i)) for i, x in enumerate(L)) for L in all_layers]
    common = sorted(set.intersection(*layer_sets))
    if not common:
        raise ValueError("No overlapping layers across paraphrases.")
    keys = ["energy","smoothness_index","spectral_entropy","hfer","fiedler_value"]
    out = []
    for L in common:
        vals_per_key = {k: [] for k in keys}
        for layers in all_layers:
            # map layer id -> record
            idx = {int(x.get("layer", i)): i for i, x in enumerate(layers)}
            rec = layers[idx[L]]
            for k in keys:
                vals_per_key[k].append(float(rec.get(k, np.nan)))
        merged = {"layer": int(L)}
        for k, arr in vals_per_key.items():
            merged[k] = float(np.nanmean(arr))
        out.append(merged)
    # sort by layer index
    out.sort(key=lambda r: r["layer"])
    return out

def main():
    ap = argparse.ArgumentParser(description="Average paraphrases into diagnostics_avg.json per (language, voice).")
    ap.add_argument("--root", required=True, help=r"Root folder like .\iclr_results\main_runs")
    ap.add_argument("--paraphrases", nargs="+", default=["p1","p2","p3"], help="Paraphrase prefixes")
    ap.add_argument("--voices", nargs="+", default=["active","passive"], help="Voices to average")
    ap.add_argument("--pattern", default="diagnostics_*.json", help="Diagnostics filename pattern")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    total_written = 0
    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for lang_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            for voice in args.voices:
                # collect latest diagnostics per paraphrase
                jsons = []
                for pfx in args.paraphrases:
                    run_dir = lang_dir / f"{pfx}_{voice}"
                    if not run_dir.exists():
                        continue
                    matches = sorted(run_dir.glob(args.pattern), key=lambda x: x.stat().st_mtime)
                    if matches:
                        jsons.append(matches[-1])
                if len(jsons) == 0:
                    if args.verbose:
                        print(f"… skip (no files): {model_dir.name}/{lang_dir.name}/{voice}")
                    continue

                try:
                    averaged = mean_across_paraphrases(jsons)
                except Exception as e:
                    print(f"⚠️  skipping {model_dir.name}/{lang_dir.name}/{voice}: {e}")
                    continue

                out_dir = lang_dir / f"avg_{voice}"
                out_path = out_dir / "diagnostics_avg.json"
                if args.dry_run:
                    print(f"[DRY] would write: {out_path}  (from {len(jsons)} files)")
                else:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"diagnostics": averaged}, f, ensure_ascii=False, indent=2)
                    total_written += 1
                    if args.verbose:
                        print(f"✅ wrote {out_path}  (from {len(jsons)} paraphrases)")

    if not args.dry_run:
        print(f"\nDone. Wrote {total_written} averaged files.")

if __name__ == "__main__":
    main()
