#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

VOICE_TYPE = {
  "en":"analytic","zh":"analytic","hi":"analytic","id":"analytic","vi":"analytic","sw":"analytic","yo":"analytic",
  "es":"periphrastic","fr":"periphrastic","it":"periphrastic","de":"periphrastic","pt":"periphrastic","ru":"periphrastic","pl":"periphrastic",
  "tr":"affixal","fi":"affixal",
  "ja":"particle","ko":"particle",
  "ar":"non-concatenative","he":"non-concatenative",
}

def to_float(x):
    """Coerce metric to a scalar: numbers -> float; lists/tuples -> mean; dicts -> value/mean/avg or mean of numeric values."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        vals = [to_float(v) for v in x]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    if isinstance(x, dict):
        for k in ("value", "mean", "avg", "median"):
            if k in x:
                return to_float(x[k])
        vals = [to_float(v) for v in x.values()]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    try:
        return float(x)
    except Exception:
        return None

def load_diag(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not (isinstance(data, dict) and "diagnostics" in data and isinstance(data["diagnostics"], list)):
        raise ValueError(f"Unexpected diagnostics format: {path}")
    return data["diagnostics"]

def average_runs(run_dirs):
    import numbers, json
    from pathlib import Path
    from statistics import mean

    per_run = []
    for rd in run_dirs:
        cand = sorted(Path(rd).glob("diagnostics_*.json"), key=lambda p: p.stat().st_mtime)
        if cand:
            per_run.append(cand[-1])
    if not per_run:
        return None

    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Accept either {"diagnostics":[...]} or plain list
        diags = obj["diagnostics"] if isinstance(obj, dict) and "diagnostics" in obj else obj
        return diags

    runs = [load(p) for p in per_run]
    if not runs or not runs[0]:
        return None

    layers = [d.get("layer") for d in runs[0]]
    # numeric scalar keys only
    def numeric_keys(sample_row):
        out = []
        for k, v in sample_row.items():
            if k == "layer": continue
            if isinstance(v, numbers.Number):
                out.append(k)
        return out

    keys = numeric_keys(runs[0][0])
    out = []
    for i, L in enumerate(layers):
        row = {"layer": int(L)}
        for k in keys:
            vals = []
            for r in runs:
                v = r[i].get(k, None)
                if isinstance(v, numbers.Number):
                    vals.append(float(v))
            row[k] = mean(vals) if vals else None
        out.append(row)
    return {"diagnostics": out}


def find_selector_dir(lang_voice_dir: Path, selector_str: str) -> Path | None:
    cand = lang_voice_dir / selector_str
    if cand.exists():
        return cand
    s = selector_str.replace("/", "\\")
    bits = [b for b in s.split("\\") if b]
    if len(bits) == 2:
        a, b = bits
        alts = {
            lang_voice_dir / a.replace("_","-") / b.replace("_","-"),
            lang_voice_dir / a.replace("-","_") / b.replace("-","_"),
            lang_voice_dir / a / b,
        }
        for dd in alts:
            if dd.exists():
                return dd
    return None

def main():
    ap = argparse.ArgumentParser(description="Average runs and build *_voice_pairs_avg.csv for a selected config.")
    ap.add_argument("--root",   required=True, help=r".\iclr_results\multi20_expanded")
    ap.add_argument("--models", nargs="+", required=True, help="e.g. qwen2.5-7b phi-3-mini llama-3.2-1b")
    ap.add_argument("--selector", required=True, help=r"e.g. 'sym-symmetric\agg-uniform' or 'sym-row_norm\agg-attention_weighted'")
    ap.add_argument("--outdir", required=True, help=r"where to write *_voice_pairs_avg.csv")
    ap.add_argument("--tmpavg", required=True, help=r"folder to save averaged diagnostics (for plotting)")
    args = ap.parse_args()

    root   = Path(args.root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tmpavg = Path(args.tmpavg); tmpavg.mkdir(parents=True, exist_ok=True)

    for mdl in args.models:
        mroot = root / mdl
        if not mroot.exists():
            print(f"⚠️  skip model (not found): {mroot}")
            continue

        found = 0
        # Average per language/voice
        for ld in sorted(mroot.glob("*_*")):
            if not ld.is_dir():
                continue
            name = ld.name
            if not (name.endswith("_active") or name.endswith("_passive")):
                continue
            lang, voice = name.split("_", 1)

            sel = find_selector_dir(ld, args.selector)
            if sel is None:
                continue
            runs = [p for p in sel.glob("run_*") if p.is_dir()]
            if not runs:
                continue

            avg = average_runs(runs)
            if avg is None:
                continue

            out_avg_dir = tmpavg / mdl / lang.lower() / voice.lower()
            out_avg_dir.mkdir(parents=True, exist_ok=True)
            with open(out_avg_dir / "diagnostics_avg.json", "w", encoding="utf-8") as f:
                json.dump(avg, f, ensure_ascii=False, indent=2)
            found += 1

        print(f"Model {mdl}: averaged {found} (lang,voice) configs.")

        # Assemble pairs
        langs = {}
        for d in (tmpavg / mdl).rglob("diagnostics_avg.json"):
            parts = d.parts
            lang = parts[-3].lower()
            voice = parts[-2].lower()
            langs.setdefault(lang, {})[voice] = str(d)

        out_csv = outdir / f"{mdl}_voice_pairs_avg.csv"
        nrows = 0
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("label,voice_type,active_json,passive_json\n")
            for lg, dct in sorted(langs.items()):
                a = dct.get("active"); p = dct.get("passive")
                if not (a and p):
                    print(f"  … skip {mdl}/{lg.upper()} (missing {'active' if not a else 'passive'})")
                    continue
                f.write(f"{lg.upper()},{VOICE_TYPE.get(lg,'unknown')},{a},{p}\n")
                nrows += 1
        print(f"✅ wrote {out_csv}  ({nrows} languages)")

if __name__ == "__main__":
    main()
