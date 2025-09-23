#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def make_per_language_compat(df: pd.DataFrame) -> pd.DataFrame:
    # continent := label (so each language is its own group)
    out = pd.DataFrame({
        "label": df["label"].astype(str),
        "continent": df["label"].astype(str),
        "active": df["active_json"].astype(str),
        "passive": df["passive_json"].astype(str),
    })
    return out

def make_by_voicetype_compat(df: pd.DataFrame) -> pd.DataFrame:
    if "voice_type" not in df.columns:
        raise SystemExit("Input CSV must have a voice_type column.")
    # continent := voice_type (so lines = voice types)
    out = pd.DataFrame({
        "label": df["label"].astype(str),               # will be ignored by plotter’s grouping
        "continent": df["voice_type"].astype(str),      # used as group label
        "active": df["active_json"].astype(str),
        "passive": df["passive_json"].astype(str),
    })
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Build plotter-compat CSVs from *_voice_pairs_avg.csv")
    ap.add_argument("--inp", required=True, help="e.g., iclr_results/manifests/qwen2.5-7b_voice_pairs_avg.csv")
    ap.add_argument("--per_language_out", required=True,
                    help="e.g., iclr_results/manifests/qwen2.5-7b_per_language_compat.csv")
    ap.add_argument("--by_voicetype_out", required=True,
                    help="e.g., iclr_results/manifests/qwen2.5-7b_by_voicetype_compat.csv")
    args = ap.parse_args()

    src = Path(args.inp)
    df = pd.read_csv(src)

    per_lang = make_per_language_compat(df)
    per_lang.to_csv(args.per_language_out, index=False)
    print(f"✓ wrote {args.per_language_out}  ({len(per_lang)} rows)")

    by_vt = make_by_voicetype_compat(df)
    by_vt.to_csv(args.by_voicetype_out, index=False)
    print(f"✓ wrote {args.by_voicetype_out}  ({len(by_vt)} rows)")

if __name__ == "__main__":
    main()
