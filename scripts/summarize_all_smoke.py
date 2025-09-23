#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path
from subprocess import run, PIPE, STDOUT
import sys

CMD = [sys.executable, str(Path(__file__).with_name("summarize_smoke_pair.py"))]

def call_pair(tag, base, ablt, out_csv, lo, hi):
    p = run(CMD + ["--baseline", base, "--ablate", ablt,
                   "--out_csv", out_csv, "--early_lo", str(lo), "--early_hi", str(hi)],
            stdout=PIPE, stderr=STDOUT, text=True)
    print(p.stdout)
    df = pd.read_csv(out_csv)
    early = df[(df.layer>=lo)&(df.layer<=hi)]["delta"].mean()
    overall = df["delta"].mean()
    return {"family": tag, "early_lo":lo, "early_hi":hi,
            "delta_lambda2_early": float(early), "delta_lambda2_overall": float(overall),
            "per_layer_csv": out_csv}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--early_lo", type=int, default=2)
    ap.add_argument("--early_hi", type=int, default=5)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--llama_base", default=r".\analysis\smoke\llama_baseline")
    ap.add_argument("--llama_ablt", default=r".\analysis\smoke\llama_headablate")
    ap.add_argument("--qwen_base",  default=r".\analysis\smoke\qwen25_baseline")
    ap.add_argument("--qwen_ablt",  default=r".\analysis\smoke\qwen25_headablate")
    ap.add_argument("--phi_base",   default=r".\analysis\smoke\phi3_baseline")
    ap.add_argument("--phi_ablt",   default=r".\analysis\smoke\phi3_headablate")
    args = ap.parse_args()

    rows=[]
    rows.append(call_pair("llama-3.2-1b", args.llama_base, args.llama_ablt, r".\analysis\smoke\numbers_llama.csv", args.early_lo, args.early_hi))
    rows.append(call_pair("qwen2.5-7b",   args.qwen_base,  args.qwen_ablt,  r".\analysis\smoke\numbers_qwen25.csv", args.early_lo, args.early_hi))
    rows.append(call_pair("phi-3-mini",   args.phi_base,   args.phi_ablt,   r".\analysis\smoke\numbers_phi3.csv", args.early_lo, args.early_hi))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n✅ wrote {out}")

if __name__ == "__main__":
    main()
