#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def zscore(x): 
    x = pd.Series(x)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def spearman_perm(x, y, n_perm=10000, seed=1234):
    """Spearman rho and permutation p-value (permute y)."""
    rng = np.random.default_rng(seed)
    x = pd.Series(x).rank(method="average").to_numpy()
    y = pd.Series(y).rank(method="average").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan, np.nan
    rho_obs = np.corrcoef(x, y)[0,1]
    cnt = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rho_perm = np.corrcoef(x, y_perm)[0,1]
        if abs(rho_perm) >= abs(rho_obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return float(rho_obs), float(p)

def ols_np(X, y):
    """OLS via numpy; returns beta, yhat, resid, R2."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] <= X.shape[1] + 1:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_tot = ((y - y.mean())**2).sum()
    ss_res = (resid**2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return beta, yhat, resid, r2, mask

def ols_perm_pvals(X, y, beta_obs, n_perm=10000, seed=1234):
    """Permutation p-values for each coefficient (permute y)."""
    rng = np.random.default_rng(seed)
    p_counts = np.zeros_like(beta_obs, dtype=int)
    n = y.size
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            beta_perm, *_ = np.linalg.lstsq(X, y_perm, rcond=None)
        except Exception:
            continue
        p_counts += (np.abs(beta_perm) >= np.abs(beta_obs)).astype(int)
    pvals = (p_counts + 1) / (n_perm + 1)
    return pvals

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="analysis/frag_delta_merged.csv")
    ap.add_argument("--outdir", required=True, help="output dir for tables/plots")
    ap.add_argument("--pieces_diff_max", type=int, default=2, help="|Δ #pieces| max for length control")
    ap.add_argument("--n_perm", type=int, default=10000, help="permutations for p-values")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.inp)

    # length control
    d0 = df.loc[df["pieces_diff"] <= args.pieces_diff_max].copy()

    # per-model z-scoring of covariates
    d0["_ppc_z"]   = d0.groupby("model")["pieces_per_char_avg"].transform(zscore)
    d0["_fragH_z"] = d0.groupby("model")["frag_entropy_avg"].transform(zscore)

    rows = []
    for mdl, d in d0.groupby("model", sort=False):
        y = d["delta_lambda2_mean"].to_numpy()

        # Spearman (univariate) with permutation p-values
        rho_ppc, p_ppc = spearman_perm(d["pieces_per_char_avg"], y, n_perm=args.n_perm, seed=args.seed)
        rho_h,   p_h   = spearman_perm(d["frag_entropy_avg"],   y, n_perm=args.n_perm, seed=args.seed+1)

        # OLS (two covariates, z-scored) + permutation p-values
        X = np.column_stack([np.ones(len(d)), d["_ppc_z"].to_numpy(), d["_fragH_z"].to_numpy()])
        ols = ols_np(X, y)
        if ols is None:
            beta = np.array([np.nan, np.nan, np.nan])
            r2 = np.nan
            pvals = np.array([np.nan, np.nan, np.nan])
        else:
            beta, yhat, resid, r2, mask = ols
            # permute only over rows used in the fit
            pvals = ols_perm_pvals(X[mask], y[mask], beta, n_perm=args.n_perm, seed=args.seed+2)

        rows.append({
            "model": mdl,
            "n_lang": len(d),
            "spearman_rho_ppc": rho_ppc, "spearman_p_ppc": p_ppc,
            "spearman_rho_fragH": rho_h, "spearman_p_fragH": p_h,
            "ols_beta_const": float(beta[0]),
            "ols_beta_ppc_z": float(beta[1]),
            "ols_beta_fragH_z": float(beta[2]),
            "ols_p_const": float(pvals[0]),
            "ols_p_ppc_z": float(pvals[1]),
            "ols_p_fragH_z": float(pvals[2]),
            "ols_r2": r2,
        })

        # scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        axes[0].scatter(d["pieces_per_char_avg"], y)
        axes[0].set_xlabel("Pieces per char (avg)")
        axes[0].set_ylabel("Δλ2 mean (layers 2–5)")
        axes[0].set_title(mdl)

        axes[1].scatter(d["frag_entropy_avg"], y)
        axes[1].set_xlabel("Frag. entropy (avg)")
        axes[1].set_ylabel("Δλ2 mean (layers 2–5)")

        # simple least squares lines (for viz only)
        for ax, xcol in zip(axes, ["pieces_per_char_avg","frag_entropy_avg"]):
            xd = d[xcol].to_numpy(); yd = y
            msk = np.isfinite(xd) & np.isfinite(yd)
            if msk.sum() >= 3:
                A = np.vstack([xd[msk], np.ones(msk.sum())]).T
                m,b = np.linalg.lstsq(A, yd[msk], rcond=None)[0]
                xs = np.linspace(xd[msk].min(), xd[msk].max(), 100)
                ax.plot(xs, m*xs + b, lw=2, alpha=0.7)

        fig.tight_layout()
        fig.savefig(outdir / f"{mdl.replace('/','_')}_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "frag_regress_summary.csv", index=False)
    d0.to_csv(outdir / "frag_delta_filtered.csv", index=False)
    print(f"✅ Wrote:\n  - {outdir/'frag_regress_summary.csv'}\n  - {outdir/'<model>_scatter.png'}\n  - {outdir/'frag_delta_filtered.csv'}")

if __name__ == "__main__":
    main()
