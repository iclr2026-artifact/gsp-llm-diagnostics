#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def paired_t(a, b):
    # simple paired t-test (no SciPy dependency)
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    n = d.size
    if n < 2: return np.nan, np.nan
    mean = d.mean()
    sd = d.std(ddof=1) if n>1 else np.nan
    t  = mean / (sd/np.sqrt(n)) if sd>0 else np.nan
    return t, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--metric", default="fiedler_auc",
                    help="fiedler_auc|fiedler_peak|fiedler_mean|fiedler_early|fiedler_late|fiedler_slope")
    ap.add_argument("--styles", nargs="+", default=["standard","cot"])
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    assert len(args.styles)==2, "use exactly two styles for paired comparison"
    sA, sB = args.styles

    for model in df['model'].unique():
        sub = df[df.model==model]
        # pivot over prompt to align pairs
        piv = sub.pivot_table(index='prompt', columns='style', values=args.metric, aggfunc='mean')
        piv = piv.dropna(subset=[sA, sB], how='any')
        t, n = paired_t(piv[sA].values, piv[sB].values)
        diff_mean = float((piv[sA] - piv[sB]).mean())
        print(f"{model:20} {sA} - {sB}  on {args.metric}: "
              f"Δmean={diff_mean:8.4f}, t≈{t:6.2f}, n={n}")

if __name__ == "__main__":
    main()
