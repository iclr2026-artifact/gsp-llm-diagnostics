#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--criterion", default="fiedler_auc", help="which summary metric to minimize")
    ap.add_argument("--out_csv", default="prompt_style_recommendations.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    grp = df.groupby(['model','prompt'])
    best = grp.apply(lambda g: g.sort_values(args.criterion, ascending=True).iloc[0]).reset_index(drop=True)
    best = best[['model','prompt','style',args.criterion,'fiedler_peak','fiedler_mean','n_layers']]
    best.to_csv(args.out_csv, index=False)
    print(f"Saved recommendation table → {args.out_csv}")

if __name__ == "__main__":
    main()
