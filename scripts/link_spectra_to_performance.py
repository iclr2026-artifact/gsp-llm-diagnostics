import argparse, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc_csv", required=True, help="accuracy_by_model_style.csv")
    ap.add_argument("--delta_csv", required=True, help="delta_vs_standard_auc.csv")
    ap.add_argument("--difficulty_csv", required=False, default=None, help="accuracy_by_difficulty.csv")
    args = ap.parse_args()

    acc = pd.read_csv(args.acc_csv)
    base = acc[acc.style=="standard"][["model","accuracy"]].rename(columns={"accuracy":"acc_standard"}).set_index("model")

    df = acc[acc.style!="standard"].merge(
        pd.read_csv(args.delta_csv),
        on=["model","style"], how="left"
    )
    df = df.join(base, on="model")
    df["acc_delta_vs_standard"] = df["accuracy"] - df["acc_standard"]

    # Simple OLS across all models/styles
    formula = "acc_delta_vs_standard ~ fiedler_auc_delta_vs_standard + entropy_auc_delta_vs_standard + energy_auc_delta_vs_standard + hfer_auc_delta_vs_standard + smi_auc_delta_vs_standard"
    model = smf.ols(formula, data=df).fit()
    print("\n=== OLS: ΔAcc ~ ΔSpectral AUCs (all models/styles) ===")
    print(model.summary())

    # Sign test: we expect fiedler coefficient negative on harder problems
    if args.difficulty_csv and Path(args.difficulty_csv).exists():
        accd = pd.read_csv(args.difficulty_csv)
        # Join difficulty accuracy with deltas by model/style
        delta = pd.read_csv(args.delta_csv)
        accd = accd.merge(delta, on=["model","style"], how="left")
        # pivot to ΔAcc vs Standard per difficulty
        std = accd[accd.style=="standard"][["model","chain_len","accuracy"]].rename(columns={"accuracy":"acc_std"})
        others = accd[accd.style!="standard"].merge(std, on=["model","chain_len"], how="left")
        others["acc_delta_vs_standard"] = others["accuracy"] - others["acc_std"]

        for L in sorted(others["chain_len"].dropna().unique()):
            subset = others[others.chain_len==L]
            mdl = smf.ols("acc_delta_vs_standard ~ fiedler_auc_delta_vs_standard + entropy_auc_delta_vs_standard + energy_auc_delta_vs_standard", data=subset).fit()
            print(f"\n=== OLS by Difficulty (chain_len={L}) ===")
            print(mdl.summary())

if __name__ == "__main__":
    main()
