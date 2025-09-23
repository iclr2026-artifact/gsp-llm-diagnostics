#!/usr/bin/env python3
import argparse, subprocess, sys, json
from pathlib import Path

STYLES = {
    "standard": "prompts/reasoning_multistep_standard.txt",
    "cot": "prompts/reasoning_multistep_cot.txt",
    "tot": "prompts/reasoning_multistep_tot.txt",
    "cod": "prompts/reasoning_multistep_chain_of_draft.txt",
}

def render_prompt(style_file: Path, question: str) -> str:
    tpl = Path(style_file).read_text(encoding="utf-8")
    # If template has a placeholder, use it; otherwise append.
    if "{QUESTION}" in tpl:
        return tpl.replace("{QUESTION}", question.strip())
    return tpl.strip() + "\n\n" + question.strip()

def run_one_file(cli_path, model, style, prompt_file, out_root, extra_flags, perlayer_csv):
    out_dir = Path(out_root) / model / style
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, cli_path, "analyze",
        "--model", model,
        "--text_file", prompt_file,
        "--output_dir", str(out_dir),
        "--save_plots",
        "--style", style,
        "--perlayer_out", str(perlayer_csv),
    ] + extra_flags
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_dataset(cli_path, models, styles, dataset_path, out_root, extra_flags, perlayer_csv):
    data = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    # Each item: { "id": "...", "question": "...", "answer": "A|B|yes|no|42" }
    for model in models:
        for style in styles:
            if style not in STYLES:
                print(f"Skipping unknown style: {style}")
                continue
            style_file = Path(STYLES[style])
            for ex in data:
                qid = str(ex.get("id", ""))
                question = ex["question"]
                gold = str(ex["answer"])
                text = render_prompt(style_file, question)

                out_dir = Path(out_root) / model / style / qid
                out_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, cli_path, "analyze",
                    "--model", model,
                    "--text", text,
                    "--gold", gold,
                    "--style", style,
                    "--qid", qid,
                    "--output_dir", str(out_dir),
                    "--save_plots",
                    "--perlayer_out", str(perlayer_csv),
                ] + extra_flags

                print(">>", " ".join(cmd[:10]), "...")  # shorten print
                subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli_path", type=str, default="cli_2.py", help="Path to your cli_2.py")
    ap.add_argument("--dataset", type=str, help="JSONL with fields: id, question, answer")
    ap.add_argument("--models", nargs="+", required=True, help="e.g. llama-3.2-1b qwen2-7b")
    ap.add_argument("--styles", nargs="+", default=["standard","cot","tot","cod"],
                    help="subset of: standard cot tot cod")
    ap.add_argument("--out_root", type=str, default="analysis/cot")
    ap.add_argument("--head_aggregation", default="attention_weighted")
    ap.add_argument("--symmetrization", default="row_norm")
    ap.add_argument("--normalization", default="sym")
    ap.add_argument("--analyzer", default="scripts/analyze_prompt_styles_all.py")
    args = ap.parse_args()

    # Single shared CSV for analyzer
    perlayer_csv = Path(args.out_root) / "prompt_style_perlayer.csv"
    perlayer_csv.parent.mkdir(parents=True, exist_ok=True)

    extra_flags = [
        "--head_aggregation", args.head_aggregation,
        "--symmetrization", args.symmetrization,
        "--normalization", args.normalization,
    ]

    if args.dataset:
        run_dataset(args.cli_path, args.models, args.styles, args.dataset,
                    args.out_root, extra_flags, perlayer_csv)
    else:
        # Old behavior: just run the static prompt files once per style
        for model in args.models:
            for style in args.styles:
                if style not in STYLES:
                    print(f"Skipping unknown style: {style}")
                    continue
                run_one_file(args.cli_path, model, style, STYLES[style],
                             args.out_root, extra_flags, perlayer_csv)

    # Kick the analyzer to compute AUCs/RCI/deltas and (optionally) plots
    summary_dir = Path(args.out_root) / "summary"
    cmd = [
        sys.executable, args.analyzer,
        "--perlayer_csv", str(perlayer_csv),
        "--out_dir", str(summary_dir),
        "--include_fiedler_in_rci",
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
