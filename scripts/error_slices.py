import argparse, json, os, glob
import pandas as pd
from pathlib import Path

def load_preds(out_root):
    rows=[]
    for p in glob.glob(os.path.join(out_root,"*","*","preds_*.jsonl")):
        model, style = Path(p).parts[-3], Path(p).parts[-2]
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line); r["model"]=model; r["style"]=style; rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--delta_csv", required=True)
    ap.add_argument("--topk", type=int, default=30)
    args=ap.parse_args()

    preds=load_preds(args.out_root)
    delta=pd.read_csv(args.delta_csv)
    # Keep CoT vs Standard deltas
    d_cot = delta[delta.style=="cot"][["model","fiedler_auc_delta_vs_standard",
                                       "entropy_auc_delta_vs_standard","energy_auc_delta_vs_standard"]]

    # Join preds for CoT + Standard on same item_id
    cot = preds[preds.style=="cot"][["model","item_id","input","gold","pred","is_correct","cot_text"]]
    std = preds[preds.style=="standard"][["model","item_id","pred","is_correct"]].rename(columns={"pred":"pred_std","is_correct":"is_correct_std"})
    merged = cot.merge(std, on=["model","item_id"], how="inner").merge(d_cot, on="model", how="left")

    # Failures where CoT is wrong but shows strong modularity (Fiedler much lower vs Standard)
    bad = merged[(merged.is_correct==False)].copy()
    bad = bad.sort_values("fiedler_auc_delta_vs_standard")  # more negative = more partitioning
    print("\n=== CoT WRONG with strongest Fiedler↓ vs Standard (top) ===")
    print(bad.head(args.topk)[["model","item_id","fiedler_auc_delta_vs_standard","energy_auc_delta_vs_standard","input","gold","pred"]].to_string(index=False))

    # Successes where Standard wins with low modularity change (Fiedler delta near 0 or positive)
    good_std = merged[(merged.is_correct_std==True) & (merged.is_correct==False)].copy()
    good_std = good_std.sort_values("fiedler_auc_delta_vs_standard", ascending=False)
    print("\n=== Standard RIGHT & CoT WRONG (FiedlerΔ near 0/positive) ===")
    print(good_std.head(args.topk)[["model","item_id","fiedler_auc_delta_vs_standard","input","gold","pred_std","pred"]].to_string(index=False))

if __name__ == "__main__":
    main()
