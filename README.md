# Early-Layer Spectral Connectivity: Model-Imprinted Signatures of Voice Processing

Training-free analysis of attention-induced token graphs in transformers. We compute the Fiedler eigenvalue λ₂ per layer, then track **early-window Δλ₂ (layers 2–5)** between passive vs. active to reveal **family-specific computational fingerprints** across 20 languages. The pipeline also includes uncertainty estimation (bootstrap + permutation + BH–FDR), tokenizer-stress correlations, targeted head ablations, and reasoning-style generalization.

## Data availability

To respect GitHub’s 1 GB repository size policy, the repo only contains the
data for **one representative language**. This is sufficient for running
the full pipeline end-to-end as a smoke test.

The **small complete dataset** (all 20 languages, ~3–7 GB depending on version)
is not included here. Interested researchers, or collaborators
can request access by contacting the authors. Upon request, we will provide
a download link (Zenodo / Hugging Face / institutional storage).

## Installation

You can set up the environment either with **pip** or **conda**.

### Option 1 - pip

```bash
# clone the repository
git clone https://github.com/iclr2026-artifact/gsp-llm-diagnostics.git
cd gsp-llm-diagnostics

# create virtual environment
python -m venv .venv
source .venv/bin/activate      # on Linux/Mac
.venv\Scripts\activate         # on Windows PowerShell

# install requirements
pip install -r requirements.txt
```

### Option 2 - conda

```bash
# clone the repository
git clone https://github.com/your-username/gsp-llm-diagnostics.git
cd gsp-llm-diagnostics

# create conda environment
conda env create -f environment.yml
conda activate gsp_llm
```

## Quick sanity check (single sentence) for GPU < 16go VRAM (slightly different results hence to be expected):

```bash
python cli_2.py compare \
  --models qwen2.5-7b phi-3-mini llama-3.2-1b  \
  --precision auto \
  --load_in_8bit
  --text "The capital of France is Paris." \
  --output_dir ./iclr_results/baseline_active

python cli_2.py compare \
  --models qwen2.5-7b phi-3-mini llama-3.2-1b \
  --text "Paris is the capital of France." \
  --precision auto \
  --load_in_8bit
  --output_dir ./iclr_results/baseline_passive
```

## Quick sanity check (single sentence) for GPU > 16go VRAM:

```bash
python cli_2.py compare \
  --models qwen2.5-7b phi-3-mini llama-3.2-1b  \
  --precision auto \
  --text "The capital of France is Paris." \
  --output_dir ./iclr_results/baseline_active

python cli_2.py compare \
  --models qwen2.5-7b phi-3-mini llama-3.2-1b \
  --text "Paris is the capital of France." \
  --precision auto \
  --output_dir ./iclr_results/baseline_passive
```

```bash
python viz.py --figure four_graph \
  --json ./iclr_results/baseline_active/comparison_results.json \
        ./iclr_results/baseline_passive/comparison_results.json \
  --output ./iclr_results/figures/active_vs_passive_four_graph.pdf

python viz.py --figure fiedler_plot \
  --json ./iclr_results/baseline_active/comparison_results.json \
        ./iclr_results/baseline_passive/comparison_results.json \
  --output ./iclr_results/figures/active_vs_passive_fiedler.pdf
```

## Main English examples

``` bash
python cli_2.py compare --models qwen2.5-7b phi-3-mini llama-3.2-1b  \
  --text "Shakespeare wrote Hamlet." \
  --precision auto \
  --output_dir ./iclr_results/shakespeare_active

python cli_2.py compare --models qwen2.5-7b phi-3-mini llama-3.2-1b  \
  --text "Hamlet was written by Shakespeare." \
  --precision auto \
  --output_dir ./iclr_results/shakespeare_passive

python viz.py --figure fiedler_plot \
  --json ./iclr_results/shakespeare_active/comparison_results.json \
        ./iclr_results/shakespeare_passive/comparison_results.json \
  --output ./iclr_results/figures/shakespeare_active_vs_passive_fiedler.pdf
  ```

## Syntactic phenomena
``` bash
python viz.py --figure four_graph \
  --json ./iclr_results/agreement_correct/comparison_results.json \
        ./iclr_results/agreement_error/comparison_results.json \
  --output ./iclr_results/figures/agreement_four_graph.pdf

python viz.py --figure four_graph \
  --json ./iclr_results/declarative/comparison_results.json \
        ./iclr_results/wh_question/comparison_results.json \
  --output ./iclr_results/figures/wh_movement_four_graph.pdf

python viz.py --figure four_graph \
  --json ./iclr_results/standard_order/comparison_results.json \
        ./iclr_results/scrambled_order/comparison_results.json \
  --output ./iclr_results/figures/word_order_four_graph.pdf
```

## Multilingual experiments
``` bash
python scripts/build_manifests_from_config.py \
  --root ./iclr_results/multi20_expanded \
  --model qwen2.5-7b \
  --selector sym-symmetric\\agg_uniform \
  --outdir ./iclr_results/manifests_symmuni/qwen2.5-7b \
  --tmpavg ./analysis/tmp_avgs

python scripts/build_grouped_manifests.py \
  --manifests ./iclr_results/manifests_symmuni

python plot_metrics.py \
  --csv ./iclr_results/manifests/qwen2.5-7b_per_language_compat.csv \
  --metric fiedler --delta --early_lo 2 --early_hi 5 \
  --show_individual \
  --out ./figures/F2_qwen_per_language_delta_fiedler.png
```

## Statistics and FDR
``` bash
python effect_analyzer.py \
  --pairs ./iclr_results/english_active_2/comparison_results.json ./iclr_results/english_passive_2/comparison_results.json \
  --pairs ./iclr_results/french/active/comparison_results.json  ./iclr_results/french/passive/comparison_results.json \
  --pairs ./iclr_results/german/active/comparison_results.json    ./iclr_results/german/passive/comparison_results.json \
  --output_dir ./analysis/voice_meta \
  --heatmaps --mask_heatmaps \
  --winsor 0.01 --trim 0.2 --fdr_q 0.05
```

## Tokenizer-stress covariates
### Run in PowerShell from the repo root. If using conda, first: conda activate gsp-llm
### If "python" isn’t on PATH, use "py" instead of "python" below.

```bash
# --- Paths (edit if your repo uses different names) ---
$DELTA      = "analysis\early_window_delta.csv"        # model,label,early_lo,early_hi,delta_lambda2_mean
$FRAG       = "analysis\frag_covariates.csv"           # model,label,voice,n_chars,n_pieces,pieces_per_char,frag_entropy
$MERGED     = "analysis\frag_delta_merged.csv"
$WITHDELTA  = "analysis\frag_delta_merged_with_deltas.csv"
$OUTDIR     = "results\tokenizer_stress"
$MANIFEST   = "data\manifests\qwen2.5-7b_by_voicetype_compat.csv"  # columns: label,continent (renamed to voice_type)

# --- Ensure output directories exist ---
$null = New-Item -ItemType Directory -Force -Path (Split-Path $MERGED)
$null = New-Item -ItemType Directory -Force -Path (Split-Path $WITHDELTA)
$null = New-Item -ItemType Directory -Force -Path $OUTDIR

# 1) Merge early-window Δλ2 with fragment covariates (adds active/passive averages & |Δpieces|)
python .\scripts\merge_frag_with_delta.py `
  --delta "$DELTA" `
  --frag  "$FRAG" `
  --out   "$MERGED"

# 2) Compute passive–active deltas (adds d_pieces_per_char, d_frag_entropy, pieces_diff_abs)
python .\scripts\frag_make_deltas.py `
  --inp "$MERGED" `
  --out "$WITHDELTA"

# 3) Regression & per-model summaries (writes tables/plots to $OUTDIR)
python .\scripts\frag_regress.py `
  --inp "$MERGED" `
  --outdir "$OUTDIR" `
  --pieces_diff_max 2 `
  --n_perm 10000 `
  --seed 1234

# 4) Spearman (per model) on delta covariates vs Δλ2 (uses WITHDELTA)
python .\scripts\frag_delta_spearman.py `
  --inp "$WITHDELTA" `
  --out "$OUTDIR\frag_delta_spearman.csv" `
  --pieces_diff_max 2 `
  --n_perm 10000

# 5) Slice by voice type (manifest maps label -> voice_type via 'continent' column)
python .\scripts\frag_by_voicetype.py `
  --inp "$WITHDELTA" `
  --manifest "$MANIFEST" `
  --out "$OUTDIR\frag_by_voicetype.csv" `
  --pieces_diff_max 2
```

## Head ablations
```bash
python cli_2.py analyze \
  --model phi-3-mini \
  --text_file ./prompts/smoke_en.txt \
  --symmetrization row_norm \
  --normalization sym \
  --head_aggregation attention_weighted \
  --ablate_heads "3:0-3; 6:5" \
  --output_dir ./analysis/smoke/phi3_headablate \
  --save_plots

python scripts/build_causal_from_smoke.py
python scripts/summarize_smoke_pair.py
python scripts/collect_smoke_numbers.py
```

## Reasoning generalization

```bash
python .\cli_2.py analyze                                                                                                                                                                                                    
    --model llama-3.2-3b-instruct `                                                            
    --input_file .\prompts\transitivity_standard.tsv `
    --style standard `                                                                                               
    --perlayer_out ".\analysis\cot\prompt_style_perlayer_4.csv" `
    --output_dir .\analysis\cot\llama-3.2-3b-instruct\standard `
    --save_plots


python scripts/summarize_prompt_styles.py --out_root ./analysis/cot

python scripts/analyze_prompt_styles_all.py \
  --perlayer_csv ./analysis/cot/prompt_style_perlayer.csv \
  --out_dir ./analysis/cot/summary \
  --include_fiedler_in_rci

```

## Robustness knobs
```batch
--normalization rw|sym

--symmetrization symmetric|row_norm|col_norm

--head_aggregation uniform|attention_weighted

--hfer_cutoff {0.1, 0.2, 0.3}
```

Early window default: 2–5; sensitivity: 1–4 and 3–6.

## Per-language and grouped plots
```bash
# Individual language deltas
python plot_metrics.py \
  --csv ./iclr_results/manifests/phi-3-mini_per_language.csv \
  --metric fiedler --delta --early_lo 2 --early_hi 5 \
  --out ./figures/phi3_per_language_delta.png

python plot_metrics.py \
  --csv ./iclr_results/manifests/llama-3.2-1b_per_language.csv \
  --metric fiedler --delta --early_lo 2 --early_hi 5 \
  --out ./figures/llama_per_language_delta.png

# Grouped by typology/continent
python plot_metrics.py \
  --csv ./iclr_results/manifests/phi-3-mini_per_language.csv \
  --metric fiedler --delta --early_lo 2 --early_hi 5 \
  --group_by typology \
  --out ./figures/phi3_grouped_typology.png
```

## Fragmentation correlations and regressions

```bash
# Leave-one-out checks
python scripts/frag_leave_one_out.py \
  --frag_csv ./analysis/frag_metrics.csv \
  --delta_csv ./analysis/delta_metrics.csv \
  --out ./analysis/frag_leave_one_out.csv

# Spearman correlations per family
python scripts/frag_delta_spearman.py \
  --frag_csv ./analysis/frag_metrics.csv \
  --delta_csv ./analysis/delta_metrics.csv \
  --out ./analysis/frag_spearman_summary.csv

# Regression models
python scripts/frag_regress.py \
  --frag_csv ./analysis/frag_metrics.csv \
  --delta_csv ./analysis/delta_metrics.csv \
  --out ./analysis/frag_regression.csv

```

## Multilingual end-to-end (20 languages)
``` bash

# Build manifests for all languages and models
python scripts/build_manifests_from_config.py \
  --root ./iclr_results/multi20_expanded \
  --model phi-3-mini \
  --selector sym-symmetric\\agg_uniform \
  --outdir ./iclr_results/manifests_symmuni/phi-3-mini \
  --tmpavg ./analysis/tmp_avgs

python scripts/build_manifests_from_config.py \
  --root ./iclr_results/multi20_expanded \
  --model llama-3.2-1b \
  --selector sym-symmetric\\agg_uniform \
  --outdir ./iclr_results/manifests_symmuni/llama-3.2-1b \
  --tmpavg ./analysis/tmp_avgs

python scripts/build_grouped_manifests.py \
  --manifests ./iclr_results/manifests_symmuni

  ```

## Reasoning signatures (Chain-of-Thought, ToT, etc.)
```bash

python scripts/summarize_prompt_styles.py \
  --prompts ./prompts/reasoning_multistep_cot.txt \
  --out_root ./analysis/reasoning

python scripts/analyze_prompt_styles_all.py \
  --perlayer_csv ./analysis/reasoning/prompt_style_perlayer.csv \
  --out_dir ./analysis/reasoning/summary \
  --include_fiedler_in_rci

```

## Causal ablations (early vs mid vs late)

```bash
# Run targeted ablations (example: Phi-3 Mini)
python cli_2.py analyze \
  --model phi-3-mini \
  --text_file ./prompts/smoke_en.txt \
  --symmetrization row_norm \
  --normalization sym \
  --head_aggregation attention_weighted \
  --ablate_heads "3:0-3; 6:5" \
  --output_dir ./analysis/smoke/phi3_headablate \
  --save_plots

# Summarize results into tables
python scripts/build_causal_from_smoke.py \
  --input ./analysis/smoke \
  --out ./analysis/causal_summary.csv

python scripts/summarize_smoke_pair.py \
  --input ./analysis/smoke \
  --out ./analysis/smoke_pair_summary.csv

python scripts/collect_smoke_numbers.py \
  --input ./analysis/smoke \
  --out ./analysis/smoke_numbers.csv

```

## Robustness knobs
```batch
--normalization rw|sym

--symmetrization symmetric|row_norm|col_norm

--head_aggregation uniform|attention_weighted

--hfer_cutoff {0.1, 0.2, 0.3}
```

Early window: default 2–5, with sensitivity runs 1–4 and 3–6.

## Reproducibility statement

Primary endpoint: Δλ₂ early [2–5], passive − active, per prompt → averaged per language.

Uncertainty: nonparametric bootstrap (2,000), permutation (10k), BH–FDR at q=0.05.

Effect sizes: trimmed Hedges’ g with winsorization (1–2%) and trimming (20%).

Tokenizer covariates: pieces/character and fragmentation entropy.

Ablations: targeted early-layer head removals; report early/mid/late windows.

Reasoning: CoT, CoD, ToT spectral signatures correlated with performance.
#

