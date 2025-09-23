#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

def detok(tokens):
    out=[]
    for t in tokens or []:
        if not isinstance(t,str): continue
        if t.startswith("Ġ") or t.startswith("▁"):
            out.append(" "+t[1:])
        else:
            out.append(t)
    s="".join(out).strip()
    s=re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    args=ap.parse_args()
    p=Path(args.diag)
    if not p.exists():
        print("", end=""); return
    data=json.loads(p.read_text(encoding="utf-8"))
    txt=(data.get("text") or "").strip()
    if not txt:
        txt=detok(data.get("tokens"))
    print(txt or "", end="")

if __name__=="__main__":
    main()
