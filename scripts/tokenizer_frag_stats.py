#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute tokenization covariates (pieces/char and fragmentation entropy)
per (model, language, voice) for the sentences you actually used.

It reads your per-language manifest CSVs (columns: label,continent,active_json,passive_json).
For each active/passive JSON path it looks for a nearby text file (default names below).
If you don't have those text files saved, supply --texts_csv with rows: model,label,voice,text.

Outputs a CSV with rows: model,label,voice,n_chars,n_pieces,pieces_per_char,frag_entropy
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

# Map your short keys to HF repo ids
MODEL_REPO = {
    "qwen2.5-7b":    "Qwen/Qwen2.5-7B",
    "phi-3-mini":    "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.2-1b":  "meta-llama/Llama-3.2-1B",
    # add more here if needed
}

# candidate filenames to discover the exact text used
CAND_TEXT_NAMES = ("text_used.txt", "prompt.txt", "input.txt", "text.txt")

def frag_entropy(pieces):
    """length-normalized entropy over subword types in one sentence"""
    if not pieces:
        return 0.0
    vals, counts = np.unique(pieces, return_counts=True)
    p = counts.astype(float) / counts.sum()
    H = -(p * np.log(p + 1e-12)).sum()
    return float(H) / float(len(pieces))

def find_text_near(json_path: Path) -> str | None:
    """Try to discover a text file near a diagnostics json path."""
    # 1) same directory
    for name in CAND_TEXT_NAMES:
        p = json_path.with_name(name)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    # 2) parent directory
    for name in CAND_TEXT_NAMES:
        p = json_path.parent / name
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    # 3) one level up (useful when avg_* dirs contain only diagnostics)
    up = json_path.parent.parent
    if up.exists():
        for name in CAND_TEXT_NAMES:
            p = up / name
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
    return None

def load_tokenizer(model_key: str):
    """Get a transformers tokenizer for a given short key."""
    repo_id = MODEL_REPO.get(model_key, model_key)  # allow passing a full repo id
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f"!! Could not load tokenizer for {model_key} ({repo_id}): {e}", file=sys.stderr)
        return None

def tokenize_pieces(tok, text: str) -> list[str]:
    """Return the list of subword tokens (no special tokens)."""
    # fast tokenizers: .tokenize returns string pieces; otherwise use encoding
    try:
        return tok.tokenize(text)
    except Exception:
        enc = tok(text, add_special_tokens=False)
        # transformers returns dict; we can re-decode ids into “pieces” if needed
        # for entropy, using ids is fine; but we prefer string tokens:
        ids = enc["input_ids"]
        try:
            return tok.convert_ids_to_tokens(ids)
        except Exception:
            # fallback to ids as string tokens
            return [str(i) for i in ids]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True,
                    help=r"Folder with *_per_language.csv (e.g., iclr_results\manifests)")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model keys (e.g., qwen2.5-7b phi-3-mini llama-3.2-1b) or HF repo ids")
    ap.add_argument("--out", required=True, help=r"Output CSV path")
    ap.add_argument("--texts_csv", default=None,
                    help="Optional CSV with columns: model,label,voice,text (used if no nearby text files)")
    args = ap.parse_args()

    mani_dir = Path(args.manifests)
    if not mani_dir.exists():
        print(f"Manifests folder not found: {mani_dir}", file=sys.stderr)
        sys.exit(1)

    # Optional external texts mapping
    texts_map = None
    if args.texts_csv:
        tdf = pd.read_csv(args.texts_csv)
        # key: (model,label,voice) -> text
        texts_map = {(str(r["model"]).lower(),
                      str(r["label"]).lower(),
                      str(r["voice"]).lower()): str(r["text"])
                     for _, r in tdf.iterrows()}

    rows = []

    for model_key in args.models:
        tok = load_tokenizer(model_key)
        if tok is None:
            print(f"-- Skipping model {model_key} (no tokenizer).", file=sys.stderr)
            continue

        src = mani_dir / f"{model_key}_per_language.csv"
        if not src.exists():
            # also try the cleaned repo_id filename
            alt = mani_dir / f"{MODEL_REPO.get(model_key, model_key).replace('/','_')}_per_language.csv"
            src = alt if alt.exists() else src

        if not src.exists():
            print(f"-- No per-language manifest for {model_key}: {src}", file=sys.stderr)
            continue

        df = pd.read_csv(src)
        for _, r in df.iterrows():
            label = str(r["label"]).lower()
            ajson = Path(r["active_json"])
            pjson = Path(r["passive_json"])

            for voice, jpath in [("active", ajson), ("passive", pjson)]:
                text = None
                # 1) external map wins if present
                if texts_map is not None:
                    text = texts_map.get((model_key.lower(), label, voice))
                # 2) otherwise try to find next to diagnostics
                if text is None:
                    text = find_text_near(Path(jpath))
                if text is None or not text.strip():
                    print(f"!! Missing text for {model_key}/{label}/{voice}; looked near {jpath}", file=sys.stderr)
                    continue

                pieces = tokenize_pieces(tok, text)
                n_chars = len(text)
                n_pieces = len(pieces)
                ppc = (n_pieces / (n_chars + 1e-6)) if n_chars > 0 else np.nan
                H = frag_entropy(pieces)

                rows.append({
                    "model": model_key,
                    "label": label.upper(),
                    "voice": voice,
                    "n_chars": n_chars,
                    "n_pieces": n_pieces,
                    "pieces_per_char": ppc,
                    "frag_entropy": H,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅ wrote {out}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
