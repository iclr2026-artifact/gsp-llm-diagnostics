#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_bar(df, by_cols, y, title, outpng):
    plt.figure(figsize=(10,6))
    # group mean with error bars
    g = df.groupby(by_cols)[y].agg(['mean','std']).reset_index()
    # simple bar by style per model (aggregate across prompts)
    for model in g['model'].unique():
        sub = g[g['model']==model]
        plt.bar([f"{model}\n{s}" for s in sub['style']], sub['mean'], yerr=sub['std'])
    plt.ylabel(y.replace('_',' ').title())
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def save_layer_curves(dfL, model, prompt, y, outpng):
    plt.figure(figsize=(8,5))
    sub = dfL[(dfL.model==model) & (dfL.prompt==prompt)]
    for style, sdat in sub.groupby('style'):
        plt.plot(sdat['layer'], sdat[y], marker='o', label=style)
    plt.xlabel("Layer")
    plt.ylabel(y.replace('_',' ').title())
    plt.title(f"{model} · {prompt}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--perlayer_csv", required=True)
    ap.add_argument("--out_dir", default=".\analysis\cot\plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dfS = pd.read_csv(args.summary_csv)
    dfL = pd.read_csv(args.perlayer_csv)

    # 2.1 Bar: average Fiedler AUC by style per model
    save_bar(dfS, by_cols=['model','style'], y='fiedler_auc',
             title="Fiedler AUC by Prompt Style (lower = less reconfiguration)",
             outpng=out_dir/"bar_fiedler_auc.png")

    # 2.2 Bar: peak Fiedler by style per model
    save_bar(dfS, by_cols=['model','style'], y='fiedler_peak',
             title="Peak Fiedler by Prompt Style (lower = smoother)",
             outpng=out_dir/"bar_fiedler_peak.png")

    # 2.3 Example layer curves: pick first prompt per model
    for model in dfL['model'].unique():
        p0 = dfL[dfL.model==model]['prompt'].iloc[0]
        save_layer_curves(dfL, model, p0, y='fiedler_value',
                          outpng=out_dir/f"layers_{model}_{p0}.png")

    print(f"Plots saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
