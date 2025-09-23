#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages_csv", required=True, help="languages20.csv with columns: code,active,passive")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model identifiers to put in the output (should match what you pass to tokenizer_frag_stats.py). "
                         "Recommend using full HF repo ids: "
                         "Qwen/Qwen2.5-7B microsoft/Phi-3-mini-4k-instruct meta-llama/Llama-3.2-1B")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g., analysis/texts_fallback.csv)")
    args = ap.parse_args()

    rows = []
    with open(args.languages_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = r["code"].strip()
            act  = r["active"].strip()
            pas  = r["passive"].strip()
            for mdl in args.models:
                rows.append({"model": mdl, "label": code.upper(), "voice": "active",  "text": act})
                rows.append({"model": mdl, "label": code.upper(), "voice": "passive", "text": pas})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","label","voice","text"])
        w.writeheader()
        w.writerows(rows)
    print(f"✅ wrote {out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
