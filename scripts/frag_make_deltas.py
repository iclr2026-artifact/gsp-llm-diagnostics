#!/usr/bin/env python3
import pandas as pd, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)   # analysis/frag_delta_merged.csv
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.inp)

# deltas (passive - active) for covariates you already have
df["d_pieces_per_char"] = df["pieces_per_char_passive"] - df["pieces_per_char_active"]
df["d_frag_entropy"]    = df["frag_entropy_passive"]    - df["frag_entropy_active"]

# keep a “tight length control” subset too
df["pieces_diff_abs"] = df["pieces_diff"].abs()
df.to_csv(args.out, index=False)
print(f"✅ wrote {args.out}  ({len(df)} rows)")
