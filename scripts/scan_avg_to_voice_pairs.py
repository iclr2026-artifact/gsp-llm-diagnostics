# scripts/scan_avg_to_voice_pairs.py
#!/usr/bin/env python3
import argparse
from pathlib import Path

VOICE_TYPE = {
  "en":"analytic","es":"periphrastic","fr":"periphrastic","it":"periphrastic","de":"periphrastic","pt":"periphrastic",
  "ja":"particle","zh":"particle","ko":"particle",
  "tr":"affixal","fi":"affixal","pl":"affixal","ru":"affixal",
  "ar":"non-concatenative","he":"non-concatenative",
  "hi":"analytic","id":"analytic","vi":"analytic","sw":"analytic","yo":"analytic",
}

def main():
    ap = argparse.ArgumentParser(description="Scan multi20_avg to build *_voice_pairs_avg.csv")
    ap.add_argument("--root", required=True, help=r".\iclr_results\multi20_avg")
    ap.add_argument("--outdir", required=True, help=r".\iclr_results\manifests_expanded")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    model_dirs = [d for d in root.iterdir() if d.is_dir()]
    if args.models:
        allow = {m.lower() for m in args.models}
        model_dirs = [d for d in model_dirs if d.name.lower() in allow]

    for mdl in model_dirs:
        rows = []
        for lang_dir in sorted(p for p in mdl.iterdir() if p.is_dir()):
            lg = lang_dir.name.lower()
            act = lang_dir / "avg_active" / "diagnostics_avg.json"
            pas = lang_dir / "avg_passive" / "diagnostics_avg.json"
            if not act.exists() or not pas.exists():
                continue
            vt = VOICE_TYPE.get(lg, "unknown")
            rows.append((lg.upper(), vt, str(act), str(pas)))
        if not rows:
            print(f"-- no rows for {mdl.name}")
            continue
        out = outdir / f"{mdl.name}_voice_pairs_avg.csv"
        with open(out, "w", encoding="utf-8") as f:
            f.write("label,voice_type,active_json,passive_json\n")
            for lab, vt, a, p in rows:
                f.write(f"{lab},{vt},{a},{p}\n")
        print(f"✅ wrote {out}  ({len(rows)} languages)")

if __name__ == "__main__":
    main()
