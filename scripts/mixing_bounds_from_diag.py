#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path
def load_lambda2(jpath):
    with open(jpath,"r",encoding="utf-8") as f:
        d=json.load(f)
    rows=d["diagnostics"] if "diagnostics" in d else d
    L=np.array([int(r.get("layer",i)) for i,r in enumerate(rows)])
    V=np.array([float(r.get("fiedler_value", np.nan)) for r in rows])
    return L, V
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--pi_min", type=float, default=1e-3, help="lower bound on stationary mass")
    ap.add_argument("--eps", type=float, default=0.1)
    args=ap.parse_args()
    L,V = load_lambda2(args.json)
    if args.layer in L:
        v = float(V[list(L).index(args.layer)])
        gap = max(1e-6, v)  # proxy for random-walk gap; conservative
        t_mix = (1.0/gap)*np.log(1.0/(args.eps*args.pi_min))
        print(f"Layer {args.layer}: λ2≈{v:.4f}, gap≈{gap:.4f}, t_mix(ε={args.eps}) ≤ {t_mix:.1f}")
    else:
        print("Layer not found.")
