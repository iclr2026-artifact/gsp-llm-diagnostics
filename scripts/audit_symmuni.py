#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r".\iclr_results\multi20_expanded")
    ap.add_argument("--model", required=True, help="qwen2.5-7b")
    ap.add_argument("--selector", default=r"sym-symmetric\agg-uniform")
    args = ap.parse_args()

    base = Path(args.root) / args.model
    cfgs = list(base.glob(rf"*_*\\{args.selector}\\run_*\\config.json"))
    print(f"Found {len(cfgs)} config(s) under {args.model}\\{args.selector}")
    bad = []
    for cfg in cfgs:
        d = json.load(open(cfg, encoding="utf-8"))
        sym = d.get("symmetrization") or d.get("config",{}).get("symmetrization")
        agg = d.get("head_aggregation") or d.get("config",{}).get("head_aggregation")
        if (sym, agg) != ("symmetric", "uniform"):
            bad.append((str(cfg), sym, agg))
    if bad:
        print("⚠️ Mismatch:")
        for path, sym, agg in bad:
            print(f"  {path} => symmetrization={sym}, head_aggregation={agg}")
    else:
        print("✅ All configs are symmetric + uniform")

if __name__ == "__main__":
    main()
