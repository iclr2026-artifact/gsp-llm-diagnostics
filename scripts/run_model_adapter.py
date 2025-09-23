# scripts/run_model_adapter.py
#!/usr/bin/env python3
import json, subprocess, hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config & Utilities
# -------------------------
CONFIG_PATH = Path("scripts/runner_config.json")

def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing {CONFIG_PATH}. Create it as shown in the instructions.")
    # BOM-tolerant read (PowerShell often writes BOM)
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _escape_for_cmd(text: str) -> str:
    """Minimal escaping for inline CLI usage; prefer {prompt} with --text_file if available."""
    t = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return t.replace('"', r'\"')

# -------------------------
# Diagnostics runner (your real pipeline)
# -------------------------
def _run_diagnostics_via_cli(cmd_template: str, model: str, prompt_path: Path, out_dir: Path, lang: str):
    """
    Template placeholders supported:
      {model}       -> model name (e.g., "qwen2.5-7b")
      {prompt}      -> path to a temp file containing the prompt
      {prompt_text} -> prompt text inlined (escaped)
      {out}         -> output directory
      {lang}        -> language code
    """
    prompt_text = prompt_path.read_text(encoding="utf-8")
    cmd = cmd_template.format(
        model=model,
        prompt=str(prompt_path),
        prompt_text=_escape_for_cmd(prompt_text),
        out=str(out_dir),
        lang=lang,
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Diagnostics command failed.\n"
            f"Exit code: {result.returncode}\n"
            f"CMD: {cmd}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

def _pick_latest_json(out_dir: Path, pattern: str) -> Optional[Path]:
    cands = sorted(out_dir.glob(pattern))
    return cands[-1] if cands else None

# -------------------------
# Behavioral score: negative mean NLL (stable scale; higher is better)
# -------------------------
_HF_CACHE: Dict[str, Any] = {}

def _get_hf(model_name: str):
    if model_name not in _HF_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        lm  = AutoModelForCausalLM.from_pretrained(model_name)
        lm.eval()
        if torch.cuda.is_available():
            lm.to("cuda")
        _HF_CACHE[model_name] = (tok, lm)
    return _HF_CACHE[model_name]

@torch.no_grad()
def _neg_mean_nll(hf_model: str, text: str) -> float:
    tok, lm = _get_hf(hf_model)
    enc = tok(text, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    out = lm(**enc, labels=enc["input_ids"])
    mean_nll = float(out.loss.item())  # average token-level NLL
    return -mean_nll  # higher is better, stable magnitude

# -------------------------
# Public API: run_model_once
# -------------------------
def run_model_once(model_name: str, prompt: str, lang: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      diagnostics: {"diagnostics":[{"layer": int, "fiedler_value": float, ...}, ...]}
      behavior: {"task":"neg_mean_nll", "score": float (higher is better)}
    """
    cfg = _load_config()
    if model_name not in cfg:
        raise RuntimeError(f"Model '{model_name}' not found in {CONFIG_PATH}.")
    entry = cfg[model_name]

    cmd_template: str = entry["cmd_template"]
    diag_glob: str    = entry.get("diag_glob", "diagnostics_*.json")
    hf_model: str     = entry["hf_model"]

    # Prepare I/O
    cache_root = Path("tmp") / "diag_cache" / model_name
    _ensure_dir(cache_root)
    key = _stable_hash(f"{lang}::{prompt}")
    out_dir = cache_root / key
    _ensure_dir(out_dir)

    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # Run your real diagnostics pipeline
    _run_diagnostics_via_cli(cmd_template, model_name, prompt_path, out_dir, lang)

    # Find produced diagnostics JSON (BOM-tolerant read)
    jp = _pick_latest_json(out_dir, diag_glob) or _pick_latest_json(out_dir, "*.json")
    if not jp:
        raise RuntimeError(f"No diagnostics JSON found in {out_dir} (pattern={diag_glob}).")

    with open(jp, "r", encoding="utf-8-sig") as f:
        diagnostics = json.load(f)

    # Behavior: negative mean NLL
    score = _neg_mean_nll(hf_model, prompt)
    behavior = {"task": "neg_mean_nll", "score": float(score)}
    return diagnostics, behavior
