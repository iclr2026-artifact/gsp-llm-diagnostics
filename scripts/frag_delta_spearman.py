#!/usr/bin/env python3
import pandas as pd, numpy as np, argparse
from pathlib import Path

def spearman_perm(x, y, n=10000, seed=1234):
    rng = np.random.default_rng(seed)
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    m = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[m], yr[m]
    if xr.size < 3: return np.nan, np.nan
    rho = np.corrcoef(xr, yr)[0,1]
    cnt = 0
    for _ in range(n):
        yp = rng.permutation(yr)
        if abs(np.corrcoef(xr, yp)[0,1]) >= abs(rho):
            cnt += 1
    return float(rho), float((cnt+1)/(n+1))

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)   # frag_delta_merged_with_deltas.csv
ap.add_argument("--out", required=True)
ap.add_argument("--pieces_diff_max", type=int, default=2)
ap.add_argument("--n_perm", type=int, default=10000)
args = ap.parse_args()

df = pd.read_csv(args.inp)
df = df.loc[df["pieces_diff_abs"] <= args.pieces_diff_max].copy()

rows = []
for mdl, g in df.groupby("model", sort=False):
    y = g["delta_lambda2_mean"].to_numpy()
    for col in ["d_pieces_per_char","d_frag_entropy"]:
        rho, p = spearman_perm(g[col], y, n=args.n_perm, seed=1234 if col=="d_pieces_per_char" else 5678)
        rows.append({"model": mdl, "covariate": col, "spearman_rho": rho, "perm_p": p, "n": len(g)})

out = Path(args.out)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"✅ wrote {out}")
