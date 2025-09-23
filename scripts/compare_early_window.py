#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd

def load_layers(json_path, key="fiedler_value"):
    d = json.load(open(json_path, encoding="utf-8"))["diagnostics"]
    L = np.array([int(x["layer"]) for x in d])
    V = np.array([float(x[key]) for x in d], float)
    return L, V

def delta(active_json, passive_json, key="fiedler_value"):
    La, Va = load_layers(active_json, key)
    Lp, Vp = load_layers(passive_json, key)
    K = np.intersect1d(La, Lp)
    ia = {int(l):i for i,l in enumerate(La)}
    ip = {int(l):i for i,l in enumerate(Lp)}
    D = Vp[[ip[int(l)] for l in K]] - Va[[ia[int(l)] for l in K]]
    return K, D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    per = {}
    for _, r in df.iterrows():
        K, D = delta(r["active"], r["passive"])
        m = D[(K >= args.early_lo) & (K <= args.early_hi)].mean()
        grp = r["continent"]  # voice type in compat files
        per.setdefault(grp, []).append(m)
    for g in sorted(per):
        vals = np.array(per[g], float)
        print(f"{g:18s}  mean={vals.mean(): .6f}  n={vals.size}")

if __name__ == "__main__":
    main()
