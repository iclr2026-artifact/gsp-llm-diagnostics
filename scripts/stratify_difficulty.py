import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", required=True, help="accuracy_with_spectral_deltas.csv from previous step")
    ap.add_argument("--per_item_csv", required=True, help="per_item_accuracy.csv from previous step")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    merged = pd.read_csv(args.preds_csv)
    per_item = pd.read_csv(args.per_item_csv)

    # Need chain_len per item. Load a small mapping:
    # Expect a CSV at prompts/transitivity_items.csv with columns: item_id, chain_len
    dif = pd.read_csv("prompts/transitivity_items.csv")  # create this if missing
    per_item = per_item.merge(dif, on="item_id", how="left")

    # Acc by model×style×difficulty
    acc_diff = per_item.groupby(["model","style","chain_len"], as_index=False)["is_correct"].mean()
    acc_diff = acc_diff.rename(columns={"is_correct":"accuracy"})

    # Print and save
    print("\n=== Accuracy by Difficulty (chain_len) ===")
    print(acc_diff.sort_values(["model","style","chain_len"]).to_string(index=False))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    acc_diff.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
