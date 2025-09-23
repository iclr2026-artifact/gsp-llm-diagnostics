#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# --- language-specific minimal flips (very conservative) ---
def flip_en(text: str) -> Optional[str]:
    # active -> passive and passive -> active for simple transitive templates
    # 1) passive -> active (e.g., "X was eaten by Y" -> "Y ate X")
    m = re.search(r"(?i)\b(.+?)\b\s+(was|were|is|are|been|being)\s+(\w+ed|\w+en)\s+by\s+(.+)", text)
    if m:
        obj, aux, part, agent = m.groups()
        # heuristic base form for -ed/-en → remove suffix; not perfect, but ok for sanity checks
        verb = re.sub(r"(ed|en)$", "", part)
        return f"{agent.strip()} {verb.strip()} {obj.strip()}"

    # 2) active -> passive (e.g., "Y ate X" -> "X was eaten by Y")
    m = re.search(r"(?i)\b(.+?)\b\s+(\w+ed|\w+en|\w+s|\w+)\s+(.+)", text)
    if m:
        subj, verb, obj = m.groups()
        # simplistic participle form
        part = verb if verb.endswith(("ed","en")) else (verb.rstrip("s") + "ed")
        return f"{obj.strip()} was {part.strip()} by {subj.strip()}"
    return None

def flip_fr(text: str) -> Optional[str]:
    # active ↔ passive via "être + participe passé … par"
    # passive -> active: "X a été V-par Y" -> "Y a V X"
    m = re.search(r"(?i)(.+?) a été ([^ ]+?) par (.+)", text)
    if m:
        obj, part, ag = m.groups()
        base = re.sub(r"(é|ée|és|ées)$", "er", part)
        return f"{ag.strip()} a {base.strip()} {obj.strip()}"
    # active -> passive: "Y a V X" -> "X a été V-é par Y"
    m = re.search(r"(?i)(.+?) a ([^ ]+?) (.+)", text)
    if m:
        subj, base, obj = m.groups()
        part = re.sub(r"er$", "é", base)
        return f"{obj.strip()} a été {part.strip()} par {subj.strip()}"
    return None

def flip_de(text: str) -> Optional[str]:
    # very rough: "Y hat X VERBt" <-> "X wird von Y VERBt"
    m = re.search(r"(?i)(.+?) hat (.+?) (\w+t)\b", text)
    if m:
        subj, obj, part = m.groups()
        return f"{obj.strip()} wird von {subj.strip()} {part.strip()}"
    m = re.search(r"(?i)(.+?) wird von (.+?) (\w+t)\b", text)
    if m:
        obj, ag, part = m.groups()
        return f"{ag.strip()} hat {obj.strip()} {part.strip()}"
    return None

def flip_es(text: str) -> Optional[str]:
    # "Y comió X" <-> "X fue comido por Y" (toy pattern)
    m = re.search(r"(?i)(.+?) fue (\w+?do|\w+?da) por (.+)", text)
    if m:
        obj, part, ag = m.groups()
        base = re.sub(r"(ado|ada|idos?|idas?)$", "ar", part)
        return f"{ag.strip()} {base.strip()} {obj.strip()}"
    m = re.search(r"(?i)(.+?) (\w+?) (.+)", text)
    if m:
        subj, verb, obj = m.groups()
        part = verb + "do"
        return f"{obj.strip()} fue {part.strip()} por {subj.strip()}"
    return None

def flip_tr(text: str) -> Optional[str]:
    # ultra-minimal: swap passive suffix -(i)l/-(i)n with active heuristic cue "tarafından"
    # "X Y tarafından V" <-> "Y X'i V-di"  (very approximate; for controlled toy sentences)
    if "tarafından" in text:
        # passive -> active: "X Y tarafından V" -> "Y X'i V-di"
        try:
            obj, rest = text.split("tarafından", 1)
            ag, verb = rest.strip().split(" ", 1)
            return f"{ag.strip()} {obj.strip()}’i {verb.strip()}"
        except Exception:
            return None
    return None

FLIPPERS = {"en": flip_en, "fr": flip_fr, "de": flip_de, "es": flip_es, "tr": flip_tr}

def tokenize_length(s: str, tok_name: Optional[str]) -> int:
    if not tok_name or AutoTokenizer is None:
        return len(s.split())
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    return len(tok.encode(s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="JSONL with fields: id, lang, text")
    ap.add_argument("--out", required=True, help="JSONL with: id, lang, orig, flipped, len_orig, len_flip, drift_ok")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name (per family); optional")
    ap.add_argument("--max_drift", type=int, default=2, help="max tokens difference allowed")
    args = ap.parse_args()

    out = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            lang = rec.get("lang","en").lower()
            txt  = rec["text"]
            flp_fn = FLIPPERS.get(lang)
            if not flp_fn: continue
            flipped = flp_fn(txt)
            if not flipped: continue
            L0 = tokenize_length(txt, args.tokenizer)
            L1 = tokenize_length(flipped, args.tokenizer)
            ok = abs(L0 - L1) <= args.max_drift
            out.append({"id":rec["id"],"lang":lang,"orig":txt,"flipped":flipped,
                        "len_orig":L0,"len_flip":L1,"drift_ok":bool(ok)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as g:
        for r in out: g.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"✅ wrote {args.out}  ({sum(int(r['drift_ok']) for r in out)} usable / {len(out)} total)")

if __name__ == "__main__":
    import json
    main()
