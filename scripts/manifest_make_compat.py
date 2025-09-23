#!/usr/bin/env python3
import argparse, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--inp", required=True)
p.add_argument("--out", required=True)
p.add_argument("--mode", choices=["per_language","by_voicetype"], required=True)
a = p.parse_args()

df = pd.read_csv(a.inp)
# Standardize column names for your plotter: label, continent, active, passive
if "active_json" in df.columns and "passive_json" in df.columns:
    df = df.rename(columns={"active_json":"active", "passive_json":"passive"})
if a.mode == "per_language":
    df["continent"] = df["label"]
elif a.mode == "by_voicetype":
    # if voice_type column exists, group by it; otherwise leave as-is
    if "voice_type" in df.columns:
        df["continent"] = df["voice_type"]
    else:
        raise SystemExit("by_voicetype mode requires a voice_type column.")
df = df[["label","continent","active","passive"]]
df.to_csv(a.out, index=False)
print(f" wrote {a.out}  ({len(df)} rows)")
