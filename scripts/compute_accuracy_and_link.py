import argparse, json, glob, os
from pathlib import Path
import pandas as pd

def read_preds(out_root):
    rows = []
    for path in glob.glob(os.path.join(out_root, "*", "*", "preds_*.jsonl")):
        model = Path(path).parts[-3]
        style = Path(path).parts[-2]
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    r["model"] = model
                    r["style"] = style
                    rows.append(r)
                except Exception:
                    pass
    if not rows:
        raise SystemExit(f"No preds_*.jsonl found under {out_root}")
    df = pd.DataFrame(rows)
    # ensure booleans
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(bool)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="e.g., ./analysis/cot")
    ap.add_argument("--spectral_delta_csv", required=False, default=None,
                    help="optional: CSV with Δ vs Standard AUCs (from analyze_prompt_styles_all.py)")
    ap.add_argument("--save_csv_dir", required=False, default=None)
    args = ap.parse_args()

    df = read_preds(args.out_root)

    # Per-model×style accuracy
    acc = df.groupby(["model","style"], as_index=False)["is_correct"].mean()
    acc = acc.rename(columns={"is_correct":"accuracy"})
    print("\n=== Accuracy by model × style ===")
    print(acc.sort_values(["model","style"]).to_string(index=False))

    # Per-item table to enable difficulty bins (needs item_id)
    if "item_id" in df.columns:
        per_item = df.groupby(["model","style","item_id"], as_index=False)["is_correct"].mean()
    else:
        per_item = None

    if args.spectral_delta_csv and os.path.exists(args.spectral_delta_csv):
        spec = pd.read_csv(args.spectral_delta_csv)
        # expected columns (as printed earlier):
        # ['model','style','energy_auc_delta_vs_standard','entropy_auc_delta_vs_standard',
        #  'hfer_auc_delta_vs_standard','smi_auc_delta_vs_standard','fiedler_auc_delta_vs_standard']
        merged = acc.merge(spec, on=["model","style"], how="left")
        print("\n=== Accuracy + Δ vs Standard (AUCs) ===")
        cols = ["model","style","accuracy",
                "fiedler_auc_delta_vs_standard","entropy_auc_delta_vs_standard",
                "energy_auc_delta_vs_standard","hfer_auc_delta_vs_standard","smi_auc_delta_vs_standard"]
        print(merged[cols].sort_values(["model","style"]).to_string(index=False))
    else:
        merged = acc

    if args.save_csv_dir:
        Path(args.save_csv_dir).mkdir(parents=True, exist_ok=True)
        acc.to_csv(Path(args.save_csv_dir)/"accuracy_by_model_style.csv", index=False)
        if per_item is not None:
            per_item.to_csv(Path(args.save_csv_dir)/"per_item_accuracy.csv", index=False)
        merged.to_csv(Path(args.save_csv_dir)/"accuracy_with_spectral_deltas.csv", index=False)

if __name__ == "__main__":
    main()
