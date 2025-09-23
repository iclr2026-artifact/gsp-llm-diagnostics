#!/usr/bin/env python3
import pandas as pd, numpy as np, argparse

def spearman_perm(x, y, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    m = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[m], yr[m]
    if xr.size < 4: return np.nan, np.nan, m.sum()
    rho = np.corrcoef(xr, yr)[0,1]
    cnt = 0
    for _ in range(n):
        yp = rng.permutation(yr)
        if abs(np.corrcoef(xr, yp)[0,1]) >= abs(rho): cnt += 1
    return float(rho), float((cnt+1)/(n+1)), m.sum()

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)   # frag_delta_merged_with_deltas.csv
ap.add_argument("--manifest", required=True) # e.g., qwen2.5-7b_by_voicetype_compat.csv (has label->voice_type)
ap.add_argument("--out", required=True)
ap.add_argument("--pieces_diff_max", type=int, default=2)
args = ap.parse_args()

df = pd.read_csv(args.inp)
mf = pd.read_csv(args.manifest)  # columns: label, continent=voice_type (compat), active, passive
mf = mf[["label","continent"]].rename(columns={"continent":"voice_type"})
df = df.merge(mf, on="label", how="left")
df = df.loc[df["pieces_diff_abs"] <= args.pieces_diff_max].copy()

rows=[]
for mdl, g1 in df.groupby("model"):
    for vt, g in g1.groupby("voice_type"):
        for col in ["d_pieces_per_char","d_frag_entropy"]:
            rho, p, n = spearman_perm(g[col], g["delta_lambda2_mean"])
            rows.append({"model":mdl,"voice_type":vt,"covariate":col,"rho":rho,"perm_p":p,"n":n})
pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"✅ wrote {args.out}")
