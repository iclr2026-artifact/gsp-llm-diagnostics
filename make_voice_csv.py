#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from glob import glob

# --------- Voice-type mapping (edit if your 20 langs differ) ----------
VOICE_TYPE = {
    # analytic (aux + be + participle): English-like
    "en": "analytic",
    # periphrastic (aux constructions, Romance/Germanic style)
    "es": "periphrastic", "fr": "periphrastic", "it": "periphrastic", "de": "periphrastic",
    "pt": "periphrastic", "ru": "periphrastic", "pl": "periphrastic", "uk": "periphrastic",
    "hi": "periphrastic", "id": "periphrastic", "sw": "periphrastic",
    # particle (e.g., Japanese)
    "ja": "particle",
    # Chinese passive marker 被 → often particle/periphrastic; we treat as particle here
    "zh": "particle",
    # affixal (morphological passive affixes)
    "tr": "affixal", "fi": "affixal",
    # non-concatenative (templatic)
    "ar": "non-concatenative", "he": "non-concatenative",
}

# --------- Helpers ----------
def _pick_json(path_dir: Path) -> Path | None:
    """
    Choose a diagnostics JSON within a directory, preferring:
      1) diagnostics_avg.json
      2) newest diagnostics_*.json
      3) newest *.json
    Returns Path or None if nothing found.
    """
    cand = path_dir / "diagnostics_avg.json"
    if cand.exists():
        return cand
    wild = sorted(path_dir.glob("diagnostics_*.json"), key=lambda p: p.stat().st_mtime)
    if wild:
        return wild[-1]
    anyjson = sorted(path_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if anyjson:
        return anyjson[-1]
    return None

def _read_layers_count(json_path: Path) -> int:
    """
    Return number of layers found in the JSON.
    Supports shapes:
      - {"diagnostics": [ { "layer": ..., ... }, ... ]}
      - {"<model>": {"layers": [ { "layer": ..., ... }, ... ]}, ...}
      - [ { "layer": ..., ... }, ... ]
    Raises on unreadable/invalid content.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # diagnostics list
    if isinstance(data, dict) and "diagnostics" in data and isinstance(data["diagnostics"], list):
        return len(data["diagnostics"])
    # comparison-like dict
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and "layers" in v and isinstance(v["layers"], list):
                return len(v["layers"])
    # raw list
    if isinstance(data, list) and data and isinstance(data[0], dict) and "layer" in data[0]:
        return len(data)
    raise ValueError(f"Unrecognized JSON shape: {json_path}")

def _norm(p: Path) -> str:
    """Normalize path for CSV (POSIX-style separators)."""
    return p.as_posix()

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Make per-model CSV of avg_active/avg_passive diagnostics for plotting."
    )
    ap.add_argument("--root", required=True, help=r"Root like .\iclr_results\main_runs")
    ap.add_argument("--outdir", required=True, help=r"Where to write CSVs")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5-7b", "phi-3-mini", "llama-3.2-1b"],
        help="Model directory names under --root",
    )
    ap.add_argument(
        "--min_langs",
        type=int,
        default=1,
        help="Require at least this many languages per model (else skip writing CSV).",
    )
    ap.add_argument(
        "--strict_avg",
        action="store_true",
        help="Require diagnostics_avg.json; if missing, do NOT fall back to diagnostics_*.json.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(2)
    outdir.mkdir(parents=True, exist_ok=True)

    exit_code = 0

    for mdl in args.models:
        mdir = root / mdl
        if not mdir.exists():
            print(f"⚠️  Skip model (not found): {mdir}")
            exit_code = max(exit_code, 1)
            continue

        langs_dirs = sorted([p for p in mdir.iterdir() if p.is_dir()])
        if not langs_dirs:
            print(f"⚠️  No language subdirs under: {mdir}")
            exit_code = max(exit_code, 1)
            continue

        rows = []
        ok_count = 0
        skipped = 0
        layer_counts = []

        for lang_dir in langs_dirs:
            lg = lang_dir.name.lower()

            act_dir = lang_dir / "avg_active"
            pas_dir = lang_dir / "avg_passive"
            if not act_dir.exists() or not pas_dir.exists():
                print(f"… skip {mdl}/{lg}: missing avg_active/avg_passive directories")
                skipped += 1
                continue

            # pick JSONs (with fallback unless strict)
            act_json = act_dir / "diagnostics_avg.json"
            pas_json = pas_dir / "diagnostics_avg.json"
            if not args.strict_avg:
                act_json = act_json if act_json.exists() else (_pick_json(act_dir) or act_json)
                pas_json = pas_json if pas_json.exists() else (_pick_json(pas_dir) or pas_json)

            if not act_json or not act_json.exists() or not pas_json or not pas_json.exists():
                print(f"… skip {mdl}/{lg}: diagnostics JSON not found (try --strict_avg off)")
                skipped += 1
                continue

            try:
                nA = _read_layers_count(act_json)
                nP = _read_layers_count(pas_json)
            except Exception as e:
                print(f"… skip {mdl}/{lg}: invalid JSON shape ({e})")
                skipped += 1
                continue

            if nA != nP:
                print(f"… skip {mdl}/{lg}: layer count mismatch (active={nA}, passive={nP})")
                skipped += 1
                continue

            vtype = VOICE_TYPE.get(lg, "unknown")
            rows.append((lg.upper(), vtype, _norm(act_json), _norm(pas_json)))
            ok_count += 1
            layer_counts.append(nA)

        if ok_count < args.min_langs:
            print(f"⚠️  {mdl}: only {ok_count} languages ≥ min_langs={args.min_langs}; NOT writing CSV.")
            exit_code = max(exit_code, 1)
            continue

        out_csv = outdir / f"{mdl.replace('/','_')}_voice_pairs_avg.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("label,voice_type,active_json,passive_json\n")
            for lab, vtype, a, p in rows:
                f.write(f"{lab},{vtype},{a},{p}\n")

        # Summary
        n_layers_str = f"layers={layer_counts[0]}" if layer_counts else "layers=?"
        if layer_counts and not all(n == layer_counts[0] for n in layer_counts):
            n_layers_str = f"layers=VAR({sorted(set(layer_counts))})"

        print(f"✅ wrote {out_csv}  (langs={ok_count}, skipped={skipped}, {n_layers_str})")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
