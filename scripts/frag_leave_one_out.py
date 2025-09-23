#!/usr/bin/env python3
import pandas as pd, numpy as np, argparse, itertools

def spearman(x, y):
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    m = np.isfinite(xr) & np.isfinite(yr)
    if m.sum() < 3: return np.nan
    return float(np.corrcoef(xr[m], yr[m])[0,1])

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--cov", choices=["d_pieces_per_char","d_frag_entropy"], required=True)
ap.add_argument("--pieces_diff_max", type=int, default=2)
args = ap.parse_args()

df = pd.read_csv(args.inp)
df = df[(df["model"]==args.model) & (df["pieces_diff_abs"]<=args.pieces_diff_max)].copy()

base = spearman(df[args.cov], df["delta_lambda2_mean"])
rows=[{"drop": "NONE", "rho": base}]
for lab in df["label"]:
    sub = df[df["label"]!=lab]
    rows.append({"drop": lab, "rho": spearman(sub[args.cov], sub["delta_lambda2_mean"])})
out = f"analysis/frag_models_deltas/loo_{args.model.replace('/','_')}_{args.cov}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("✅ wrote", out)
