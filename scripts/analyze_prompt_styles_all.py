# -*- coding: utf-8 -*-
"""
Analyze prompt-style diagnostics across models in one go.

- Loads a per-layer CSV with columns:
  model,style,layer,fiedler_value,energy,smoothness_index,spectral_entropy,hfer,(prediction,gold,is_correct,qid,source optional)
- Aggregates AUC + early/late means
- Computes deltas vs Standard
- Builds a Reconfiguration Cost Index (RCI) per model/style
- Computes accuracy per model/style
- Exports CSVs + LaTeX + per-model plots

Usage:
  python analyze_prompt_styles_all.py \
    --perlayer_csv ./analysis/cot/prompt_style_perlayer.csv \
    --out_dir ./analysis/cot/summary \
    --include_fiedler_in_rci
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# I/O utils you already had
# ------------------------------
class PredLogger:
    def __init__(self, out_dir: str, model: str, style: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / f"predictions_{model}_{style}.jsonl"
        self.f = open(self.path, "w", encoding="utf-8")
    def log(self, record: dict):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "model","style","layer",
        "fiedler_value","energy","smoothness_index","spectral_entropy","hfer"
    }
    # case-insensitive rename
    cols_lc = {c.lower(): c for c in df.columns}
    missing = required - set(cols_lc.keys())
    if missing:
        raise RuntimeError(f"CSV missing required columns. Need: {sorted(list(required))}")

    df = df.rename(columns={cols_lc["fiedler_value"]: "fiedler_value",
                            cols_lc["energy"]: "energy",
                            cols_lc["smoothness_index"]: "smoothness_index",
                            cols_lc["spectral_entropy"]: "spectral_entropy",
                            cols_lc["hfer"]: "hfer",
                            cols_lc["model"]: "model",
                            cols_lc["style"]: "style",
                            cols_lc["layer"]: "layer"})
    # enforce dtypes
    df["layer"] = df["layer"].astype(int)
    df["model"] = df["model"].astype(str)
    df["style"] = df["style"].astype(str).str.lower()

    # Soft-normalize optional columns if present
    for opt in ["is_correct", "prediction", "gold", "qid", "sample_id", "prompt_id", "uid", "source"]:
        if opt in df.columns:
            # keep as is, but coerce bool-ish for is_correct
            if opt == "is_correct":
                df["is_correct"] = df["is_correct"].astype(float)  # 1/0 or NaN ok
    return df

# ------------------------------
# Aggregations you already had
# ------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    def agg_one(g):
        g = g.sort_values("layer")
        L = len(g); k = max(1, L // 3)
        out = {}
        out["fiedler_auc"] = g["fiedler_value"].sum()
        out["energy_auc"]  = g["energy"].sum()
        out["entropy_auc"] = g["spectral_entropy"].sum()
        out["hfer_auc"]    = g["hfer"].sum()
        out["smi_auc"]     = g["smoothness_index"].sum()
        out["fiedler_early"] = g["fiedler_value"].iloc[:k].mean()
        out["fiedler_late"]  = g["fiedler_value"].iloc[-k:].mean()
        out["entropy_early"] = g["spectral_entropy"].iloc[:k].mean()
        out["entropy_late"]  = g["spectral_entropy"].iloc[-k:].mean()
        out["hfer_early"]    = g["hfer"].iloc[:k].mean()
        out["hfer_late"]     = g["hfer"].iloc[-k:].mean()
        out["smi_early"]     = g["smoothness_index"].iloc[:k].mean()
        out["smi_late"]      = g["smoothness_index"].iloc[-k:].mean()
        return pd.Series(out)
    summ = df.groupby(["model","style"], as_index=False).apply(agg_one).reset_index(drop=True)
    return summ.sort_values(["model","style"])

def deltas_vs_standard(summ: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    assert "model" in summ.columns and "style" in summ.columns
    summ = summ.copy()
    summ["style"] = summ["style"].astype(str).str.lower()
    base = (
        summ.loc[summ["style"] == "standard", ["model"] + columns]
            .drop_duplicates(subset=["model"]).set_index("model")
    )
    rows = []
    others = summ.loc[summ["style"] != "standard", ["model", "style"] + columns]
    for _, r in others.iterrows():
        m, st = r["model"], r["style"]
        if m not in base.index:
            continue
        delta_vals = r[columns] - base.loc[m, columns]
        row = {"model": m, "style": st}
        for c in columns:
            row[f"{c}_delta_vs_standard"] = float(delta_vals[c])
        rows.append(row)
    return pd.DataFrame(rows)

def zscore_per_model(summ: pd.DataFrame, cols: list) -> pd.DataFrame:
    zs = []
    for m, g in summ.groupby("model"):
        gg = g.copy()
        for c in cols:
            mu, sd = gg[c].mean(), gg[c].std(ddof=0)
            gg[c+"_z"] = (gg[c] - mu) / sd if sd and np.isfinite(sd) else 0.0
        zs.append(gg)
    return pd.concat(zs, ignore_index=True)

def compute_rci(summ: pd.DataFrame, include_fiedler: bool) -> pd.DataFrame:
    summ_z = zscore_per_model(summ, ["energy_auc","entropy_auc","hfer_auc","smi_auc","fiedler_auc"])
    summ_z["smi_auc_cost_z"] = -summ_z["smi_auc_z"]
    components = ["energy_auc_z","entropy_auc_z","hfer_auc_z","smi_auc_cost_z"]
    if include_fiedler and "fiedler_auc_z" in summ_z.columns:
        components.append("fiedler_auc_z")
    summ_z["RCI"] = summ_z[components].mean(axis=1)
    cols = ["model","style","RCI"] + components
    return summ_z[cols].sort_values(["model","RCI"])

def plot_trajectories(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["spectral_entropy","hfer","smoothness_index"]
    for model, g in df.groupby("model"):
        for metric in metrics:
            pivot = g.pivot_table(index="layer", columns="style", values=metric, aggfunc="mean")
            ax = pivot.plot(figsize=(7.5,4.5), title=f"{model} — {metric} by layer")
            ax.set_xlabel("Layer"); ax.set_ylabel(metric.replace("_"," ").title())
            plt.tight_layout()
            p = out_dir / f"{model.replace('/','_')}_{metric}_by_layer.png"
            plt.savefig(p, dpi=180); plt.close()

# ------------------------------
# NEW: Accuracy computation
# ------------------------------
def _pick_sample_key(df: pd.DataFrame) -> str | None:
    for k in ["qid", "sample_id", "prompt_id", "uid", "source"]:
        if k in df.columns:
            return k
    return None

def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy per (model, style).
    Expects 'is_correct' column (0/1 or bool).
    Uses a sample identifier to avoid counting all layers per sample:
      - Prefer 'qid' (recommended), else sample_id/prompt_id/uid, else 'source'.
      - If none exists, falls back to taking the last layer per (model, style) group,
        which assumes 'is_correct' is constant across layers within a sample.
    """
    if "is_correct" not in df.columns:
        # No accuracy available
        return pd.DataFrame(columns=["model","style","accuracy","n"])

    df2 = df.copy()
    key = _pick_sample_key(df2)

    if key:
        # Reduce to one row per sample: take the max layer row per (model,style,key)
        if "layer" in df2.columns:
            idx = df2.groupby(["model","style",key])["layer"].transform("max") == df2["layer"]
            red = df2.loc[idx, ["model","style",key,"is_correct"]].drop_duplicates(["model","style",key])
        else:
            red = df2[["model","style",key,"is_correct"]].drop_duplicates(["model","style",key])
        acc = (
            red.groupby(["model","style"])
               .agg(accuracy=("is_correct", "mean"), n=(key, "nunique"))
               .reset_index()
               .sort_values(["model","style"])
        )
        return acc

    # Fallback: assume const is_correct per sample; pick last layer row per (model,style) block
    # This is crude if multiple samples are mixed without IDs, but better than averaging across layers.
    tmp = df2.copy()
    if "layer" in tmp.columns:
        # keep per (model,style) last layer only
        last = tmp.groupby(["model","style"])["layer"].transform("max") == tmp["layer"]
        tmp = tmp.loc[last]
    acc = (
        tmp.groupby(["model","style"])
           .agg(accuracy=("is_correct", "mean"),
                n=("is_correct", "count"))
           .reset_index()
           .sort_values(["model","style"])
    )
    return acc

# ------------------------------
# NEW: Accuracy plots
# ------------------------------
def plot_accuracy_bars(acc: pd.DataFrame, out_dir: Path):
    if acc.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # pivot: rows=model, cols=style, values=accuracy
    piv = acc.pivot_table(index="model", columns="style", values="accuracy", aggfunc="mean")
    ax = piv.plot(kind="bar", figsize=(8.5,5), rot=0, ylim=(0,1),
                  title="Accuracy by prompt style (↑ better)")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_bars.png", dpi=180)
    plt.close()

def plot_accuracy_vs_rci(acc: pd.DataFrame, rci: pd.DataFrame, out_dir: Path):
    if acc.empty or rci.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    mr = pd.merge(acc, rci[["model","style","RCI"]], on=["model","style"], how="inner")
    if mr.empty:
        return
    # One scatter with labels per point (model:style)
    plt.figure(figsize=(7.5,5))
    plt.scatter(mr["RCI"], mr["accuracy"])
    for _, r in mr.iterrows():
        plt.annotate(f"{r['model']}:{r['style']}", (r["RCI"], r["accuracy"]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.xlabel("RCI (lower = less reconfiguration)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs RCI (per model × style)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_rci.png", dpi=180)
    plt.close()

# ------------------------------
# Save tables (extended)
# ------------------------------
def save_tables(out_dir: Path,
                summ: pd.DataFrame,
                deltas: pd.DataFrame,
                rci: pd.DataFrame,
                acc: pd.DataFrame):  # NEW
    out_dir.mkdir(parents=True, exist_ok=True)

    summ_csv   = out_dir / "summary_agg.csv"
    deltas_csv = out_dir / "deltas_vs_standard.csv"
    rci_csv    = out_dir / "rci_table.csv"
    acc_csv    = out_dir / "accuracy_table.csv"  # NEW

    summ.to_csv(summ_csv, index=False)
    deltas.to_csv(deltas_csv, index=False)
    rci.to_csv(rci_csv, index=False)
    acc.to_csv(acc_csv, index=False)             # NEW

    # LaTeX (unchanged + a short accuracy table)
    try:
        auc_delta_cols = ["model","style","fiedler_auc","entropy_auc","hfer_auc","smi_auc","energy_auc"]
        latex_deltas = deltas[auc_delta_cols].round(3).to_latex(index=False, escape=False)
    except Exception:
        latex_deltas = "% Deltas table unavailable (missing columns)\n"
    (out_dir / "table_deltas.tex").write_text(latex_deltas, encoding="utf-8")

    try:
        latex_rci = rci[["model","style","RCI"]].round(3).to_latex(index=False, escape=False)
    except Exception:
        latex_rci = "% RCI table unavailable\n"
    (out_dir / "table_rci.tex").write_text(latex_rci, encoding="utf-8")

    try:
        latex_acc = acc[["model","style","accuracy","n"]].round(3).to_latex(index=False, escape=False)
    except Exception:
        latex_acc = "% Accuracy table unavailable\n"
    (out_dir / "table_accuracy.tex").write_text(latex_acc, encoding="utf-8")

    manifest = {
        "summary_csv": str(summ_csv),
        "deltas_csv": str(deltas_csv),
        "rci_csv": str(rci_csv),
        "accuracy_csv": str(acc_csv),
        "latex_deltas": str(out_dir / "table_deltas.tex"),
        "latex_rci": str(out_dir / "table_rci.tex"),
        "latex_accuracy": str(out_dir / "table_accuracy.tex"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Summarize prompt-style diagnostics and compute RCI + Accuracy.")
    ap.add_argument("--perlayer_csv", type=Path, required=True,
                    help="Path to per-layer CSV (e.g., ./analysis/cot/prompt_style_perlayer.csv)")
    ap.add_argument("--out_dir", type=Path, default=Path("./analysis/cot/summary"),
                    help="Output directory for tables/plots.")
    ap.add_argument("--include_fiedler_in_rci", action="store_true",
                    help="If set, include Fiedler AUC (z-scored) as a component in RCI.")
    ap.add_argument("--no_plots", action="store_true", help="Skip plot generation.")
    args = ap.parse_args()

    print(f"[INFO] Loading {args.perlayer_csv}")
    df = load_data(args.perlayer_csv)

    print("[INFO] Aggregating metrics (AUCs + early/late means)...")
    summ = aggregate(df)

    print("[INFO] Computing deltas vs Standard...")
    delta_cols = ["fiedler_auc","entropy_auc","hfer_auc","smi_auc","energy_auc",
                  "fiedler_early","entropy_early","hfer_early","smi_early"]
    deltas = deltas_vs_standard(summ, delta_cols)

    print("[INFO] Computing RCI per model/style "
          f"(include_fiedler={bool(args.include_fiedler_in_rci)})...")
    rci = compute_rci(summ, include_fiedler=args.include_fiedler_in_rci)

    # --- NEW: Accuracy ---
    print("[INFO] Computing accuracy per model/style...")
    acc = compute_accuracy(df)

    print("[INFO] Saving tables...")
    save_tables(args.out_dir, summ, deltas, rci, acc)

    if not args.no_plots:
        print("[INFO] Saving per-model trajectories...")
        plot_trajectories(df, args.out_dir / "plots")

        print("[INFO] Plotting accuracy bars...")
        plot_accuracy_bars(acc, args.out_dir / "plots")

        print("[INFO] Plotting accuracy vs RCI...")
        plot_accuracy_vs_rci(acc, rci, args.out_dir / "plots")

    # Console quick looks
    print("\n=== RCI (lower = less reconfiguration) ===")
    print((rci.round(3)).to_string(index=False))

    if not acc.empty:
        print("\n=== Accuracy by style ===")
        print((acc.sort_values(["model","style"]).round(3)).to_string(index=False))
    else:
        print("\n[WARN] No 'is_correct' column found; accuracy skipped.")

    if not deltas.empty:
        print("\n=== Δ vs Standard (AUCs only) ===")
        base_auc_cols = ["energy_auc", "entropy_auc", "hfer_auc", "smi_auc"]
        if args.include_fiedler_in_rci:
            base_auc_cols.append("fiedler_auc")
        delta_cols_print = [f"{c}_delta_vs_standard" for c in base_auc_cols]
        present = [c for c in delta_cols_print if c in deltas.columns]
        if present:
            cols_to_print = ["model", "style"] + present
            tbl = deltas[cols_to_print].sort_values(["model", "style"]).round(3)
            print(tbl.to_string(index=False))
            deltas_out = os.path.join(args.out_dir, "delta_auc_vs_standard.csv")
            tbl.to_csv(deltas_out, index=False)
            print(f"[INFO] Δ AUC table saved to {deltas_out}")
        else:
            print("[WARN] No delta AUC columns present to display.")
    else:
        print("\n[WARN] No non-standard styles found for delta computation.")

    print(f"\n[DONE] Outputs in: {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
