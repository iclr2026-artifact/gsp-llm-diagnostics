#!/usr/bin/env python3
import argparse, pandas as pd

# Map short keys <-> HF repo ids (case-insensitive on input)
MODEL_ALIAS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
}

def norm_model(m):
    if pd.isna(m): return m
    s = str(m).strip()
    # already looks like HF? keep it
    if "/" in s: return s
    # short key -> HF id (case-insensitive)
    return MODEL_ALIAS.get(s.lower(), s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True, help="analysis/early_window_delta.csv")
    ap.add_argument("--frag", required=True, help="analysis/frag_covariates.csv (active+passive rows)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.delta)  # columns: model,label,early_lo,early_hi,delta_lambda2_mean
    f = pd.read_csv(args.frag)   # columns: model,label,voice,n_chars,n_pieces,pieces_per_char,frag_entropy

    # normalize model ids on BOTH dataframes
    d["model"] = d["model"].map(norm_model)
    f["model"] = f["model"].map(norm_model)

    # make labels consistent (upper)
    d["label"] = d["label"].str.upper()
    f["label"] = f["label"].str.upper()

    # pivot frag to wide so we can compute active/passive averages & diffs
    pivot = f.pivot_table(
        index=["model","label"], columns="voice",
        values=["n_chars","n_pieces","pieces_per_char","frag_entropy"], aggfunc="first"
    )
    pivot.columns = [f"{a}_{b}" for a,b in pivot.columns]  # e.g., pieces_per_char_active
    pivot = pivot.reset_index()

    # sanity: ensure required columns exist
    needed = [
        "n_pieces_active","n_pieces_passive",
        "pieces_per_char_active","pieces_per_char_passive",
        "frag_entropy_active","frag_entropy_passive",
    ]
    missing = [c for c in needed if c not in pivot.columns]
    if missing:
        raise SystemExit(f"Missing expected columns after pivot: {missing}. "
                         f"Check that 'voice' has both 'active' and 'passive' rows per (model,label).")

    # covariates
    pivot["pieces_diff"] = (pivot["n_pieces_passive"] - pivot["n_pieces_active"]).abs()
    pivot["pieces_per_char_avg"] = (pivot["pieces_per_char_active"] + pivot["pieces_per_char_passive"]) / 2.0
    pivot["frag_entropy_avg"]    = (pivot["frag_entropy_active"] + pivot["frag_entropy_passive"]) / 2.0

    merged = d.merge(pivot, on=["model","label"], how="inner")
    merged.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out}  ({len(merged)} rows)")
    if len(merged) == 0:
        print("⚠️ Merge returned 0 rows. Double-check model ids and labels in both inputs.")

if __name__ == "__main__":
    main()
