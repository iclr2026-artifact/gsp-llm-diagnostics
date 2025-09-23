#!/usr/bin/env python3
"""
Layer-Specific Statistical Analysis of Syntactic Effects in Spectral Signatures
Robust, significance-aware analysis with symmetric % change, winsorization,
trimmed Hedges' g, paired Wilcoxon tests, and FDR correction.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Robust helpers
# =========================
def winsorize(a, p=0.01):
    a = np.asarray(a, dtype=float)
    if a.size == 0: return a
    lo, hi = np.quantile(a, p), np.quantile(a, 1 - p)
    return np.clip(a, lo, hi)

def symmetric_pct_change(x, y, eps):
    """
    Robust bounded percent-change: s% = 200 * (x - y) / max(x + y, eps)
    Bounded to [-200, 200] per item.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    den = np.maximum(x + y, eps)
    return 200.0 * (x - y) / den

def hedges_g(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0: return np.nan
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if sp == 0: return 0.0
    d = (mx - my) / sp
    J = 1 - (3 / max(4 * (nx + ny) - 9, 1))
    return J * d

def trimmed_hedges_g(x, y, trim=0.2, winsor_p=0.1):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0: return np.nan
    # Trimmed means
    def tmean(a):
        if len(a) < 2: return float(np.mean(a))
        a_sorted = np.sort(a)
        k = int(np.floor(trim * len(a_sorted)))
        a_slice = a_sorted[k:len(a_sorted)-k] if len(a_sorted) - 2*k > 0 else a_sorted
        return float(np.mean(a_slice))
    xm = tmean(x); ym = tmean(y)
    # Winsorized pooled SD
    xw = winsorize(x, winsor_p); yw = winsorize(y, winsor_p)
    sp = np.sqrt((np.var(xw, ddof=1) + np.var(yw, ddof=1)) / 2.0)
    if sp == 0: return 0.0
    return (xm - ym) / sp

def paired_wilcoxon_p(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:  # need a few layers
        return 1.0
    try:
        stat, p = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', correction=True, mode='auto')
        return float(p)
    except ValueError:
        return 1.0

def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    cutoff = ranked[k - 1] if k > 0 else 0.0
    mask = p <= cutoff
    # Unsort back to original order
    unsort = np.empty_like(mask)
    unsort[order] = mask
    return unsort, cutoff

class LayerSpecificSyntacticAnalyzer:
    def __init__(self, winsor_p=0.01, trim=0.2, alpha=0.05, fdr_q=0.05, robust=True, mask_heatmaps=True):
        self.metrics = ["energy", "hfer", "spectral_entropy", "smoothness_index", "fiedler_value"]
        self.model_labels = {
            "phi-3-mini": "Phi-3 Mini",
            "llama-3.2-1b": "LLaMA 3.2 1B",
            "qwen2.5-7b": "Qwen2.5 7B",
        }
        self.winsor_p = winsor_p
        self.trim = trim
        self.alpha = alpha
        self.fdr_q = fdr_q
        self.robust = robust
        self.mask_heatmaps = mask_heatmaps

    def load_json_data(self, filepath):
        """Load and normalize JSON data"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Normalize model names
        normalized = {}
        for key, value in data.items():
            key_norm = key.lower().replace('-', '').replace('_', '').replace(' ', '')
            if key_norm == 'phi3mini':
                normalized['phi-3-mini'] = value
            elif 'llama' in key.lower() and ('3.2' in key or '32' in key) and '1b' in key.lower():
                normalized['llama-3.2-1b'] = value
            elif 'qwen' in key.lower() and ('2.5' in key or '25' in key) and '7b' in key.lower():
                normalized['qwen2.5-7b'] = value
            else:
                normalized[key] = value
        return normalized

    def extract_condition_name(self, filepath):
        path = Path(filepath)
        condition = path.parent.name
        condition = condition.replace("_", " ").replace("-", " ")
        condition = condition.replace("iclr results", "").strip()
        return condition.title() if condition else path.stem

    def _get_layer_series(self, bundle, metric):
        """Return per-layer series for a metric (list of floats)."""
        layers = bundle.get('layers', [])
        vals = []
        for L in layers:
            v = L.get(metric)
            if v is None: return []
            vals.append(float(v))
        return vals

    def analyze_layer_wise_effects(self, baseline_data, comparison_data, model, metric):
        """Analyze effects per model/metric across layers with robust stats."""
        if model not in baseline_data or model not in comparison_data:
            return {}

        b_series = self._get_layer_series(baseline_data[model], metric)
        c_series = self._get_layer_series(comparison_data[model], metric)
        if not b_series or not c_series:
            return {}

        L = min(len(b_series), len(c_series))
        b = np.asarray(b_series[:L], dtype=float)
        c = np.asarray(c_series[:L], dtype=float)

        # Winsorize across layers to tame outliers
        if self.robust:
            bw = winsorize(b, self.winsor_p)
            cw = winsorize(c, self.winsor_p)
        else:
            bw, cw = b, c

        # Per-pair epsilon floor (5th percentile of (b+c))
        eps = np.quantile(bw + cw, 0.05) if self.robust else 1e-9
        s_pct = symmetric_pct_change(cw, bw, eps)  # comparison vs baseline, per layer
        floor_hits_frac = float(np.mean((bw + cw) < eps))

        # Max effects (abs) across layers
        abs_diff = np.abs(c - b)
        max_abs_diff = float(np.max(abs_diff))
        max_abs_diff_layer = int(np.argmax(abs_diff))

        abs_s_pct = np.abs(s_pct)
        max_s_pct = float(np.max(abs_s_pct))
        max_s_pct_layer = int(np.argmax(abs_s_pct))

        # Early (layers 0..min(10,L-1)), Final (last 5 or up to L)
        early_end = min(10, L - 1)
        early_slice = slice(0, early_end + 1) if L > 0 else slice(0, 0)
        final_slice = slice(max(L - 5, 0), L)

        b_early, c_early = bw[early_slice], cw[early_slice]
        b_final, c_final = bw[final_slice], cw[final_slice]

        # Effect sizes
        g_overall = hedges_g(bw, cw)
        g_early = hedges_g(b_early, c_early)
        g_final = hedges_g(b_final, c_final)

        gtrim_overall = trimmed_hedges_g(bw, cw, trim=self.trim, winsor_p=0.1)
        gtrim_early = trimmed_hedges_g(b_early, c_early, trim=self.trim, winsor_p=0.1)
        gtrim_final = trimmed_hedges_g(b_final, c_final, trim=self.trim, winsor_p=0.1)

        # Paired Wilcoxon across layers
        p_overall = paired_wilcoxon_p(cw, bw)
        p_early = paired_wilcoxon_p(c_early, b_early)
        p_final = paired_wilcoxon_p(c_final, b_final)

        return {
            'overall_g': g_overall,
            'overall_g_trim': gtrim_overall,
            'overall_p': p_overall,
            'early_g': g_early,
            'early_g_trim': gtrim_early,
            'early_p': p_early,
            'final_g': g_final,
            'final_g_trim': gtrim_final,
            'final_p': p_final,
            'max_absolute_diff': max_abs_diff,
            'max_absolute_diff_layer': max_abs_diff_layer,
            'max_symmetric_pct': max_s_pct,
            'max_symmetric_pct_layer': max_s_pct_layer,
            'baseline_mean': float(np.mean(b)),
            'comparison_mean': float(np.mean(c)),
            'baseline_std': float(np.std(b)),
            'comparison_std': float(np.std(c)),
            'total_layers': int(L),
            'floor_hits_frac': floor_hits_frac,
        }

    def analyze_phenomenon_pair_layerwise(self, baseline_file, comparison_file):
        """Analyze effect sizes with layer-specific focus"""
        baseline_data = self.load_json_data(baseline_file)
        comparison_data = self.load_json_data(comparison_file)

        baseline_name = self.extract_condition_name(baseline_file)
        comparison_name = self.extract_condition_name(comparison_file)

        results = []
        common_models = set(baseline_data.keys()) & set(comparison_data.keys())

        for model in common_models:
            for metric in self.metrics:
                layer_analysis = self.analyze_layer_wise_effects(
                    baseline_data, comparison_data, model, metric
                )
                if layer_analysis:
                    results.append({
                        'Model': self.model_labels.get(model, model),
                        'Metric': metric.replace('_', ' ').title(),
                        'Baseline_Condition': baseline_name,
                        'Comparison_Condition': comparison_name,
                        'Phenomenon': f"{baseline_name} vs {comparison_name}",
                        'Overall_g': layer_analysis['overall_g'],
                        'Overall_g_trim': layer_analysis['overall_g_trim'],
                        'Overall_p': layer_analysis['overall_p'],
                        'Early_Layers_g': layer_analysis['early_g'],
                        'Early_Layers_g_trim': layer_analysis['early_g_trim'],
                        'Early_Layers_p': layer_analysis['early_p'],
                        'Final_Layers_g': layer_analysis['final_g'],
                        'Final_Layers_g_trim': layer_analysis['final_g_trim'],
                        'Final_Layers_p': layer_analysis['final_p'],
                        'Max_Absolute_Diff': layer_analysis['max_absolute_diff'],
                        'Max_Absolute_Diff_Layer': layer_analysis['max_absolute_diff_layer'],
                        'Max_Symmetric_%_Change': layer_analysis['max_symmetric_pct'],
                        'Max_Symmetric_%_Change_Layer': layer_analysis['max_symmetric_pct_layer'],
                        'Baseline_Mean': layer_analysis['baseline_mean'],
                        'Comparison_Mean': layer_analysis['comparison_mean'],
                        'Baseline_Std': layer_analysis['baseline_std'],
                        'Comparison_Std': layer_analysis['comparison_std'],
                        'Total_Layers': layer_analysis['total_layers'],
                        'Floor_Hits_Frac': layer_analysis['floor_hits_frac'],
                    })

        return pd.DataFrame(results)

    def _classify_effect(self, g):
        if np.isnan(g): return 'N/A'
        a = abs(g)
        if a > 1.2: return 'Very Large'
        if a > 0.8: return 'Large'
        if a > 0.5: return 'Medium'
        if a > 0.2: return 'Small'
        return 'Negligible'

    def create_dramatic_effects_summary(self, df, save_path=None, alpha=0.05, fdr_q=0.05):
        """Flag dramatic effects with significance and robustness."""
        if df.empty:
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(save_path, index=False)
            print("Found 0 dramatic effects (no data).")
            return pd.DataFrame()

        # FDR over the early-layer p-values (primary)
        pvals = df['Early_Layers_p'].fillna(1.0).values
        fdr_mask, cutoff = bh_fdr(pvals, q=fdr_q)
        df = df.copy()
        df['FDR_Significant'] = fdr_mask
        df['FDR_Cutoff'] = cutoff

        # Dramatic criteria (edit if desired):
        #  - |Early g_trim| >= 0.3 (non-trivial)
        #  - Max symmetric % change >= 100
        #  - FDR significant
        #  - Not floor dominated
        dramatic_cases = df[
            (df['FDR_Significant']) &
            (df['Early_Layers_g_trim'].abs() >= 0.3) &
            (df['Max_Symmetric_%_Change'].abs() >= 100.0) &
            (df['Floor_Hits_Frac'] < 0.2)
        ].copy()

        if len(dramatic_cases) > 0:
            print(f"Found {len(dramatic_cases)} dramatic effects:")
            for _, row in dramatic_cases.iterrows():
                print(f"  {row['Model']} - {row['Phenomenon']} - {row['Metric']}")
                print(f"    Early layers g_trim: {row['Early_Layers_g_trim']:.3f}")
                print(f"    Max symmetric change: {row['Max_Symmetric_%_Change']:.1f}% at layer {int(row['Max_Symmetric_%_Change_Layer'])}")
                print()

        summary_cols = [
            'Model', 'Phenomenon', 'Metric',
            'Early_Layers_g_trim', 'Final_Layers_g_trim',
            'Early_Layers_p', 'Final_Layers_p',
            'FDR_Significant', 'FDR_Cutoff',
            'Max_Symmetric_%_Change', 'Max_Symmetric_%_Change_Layer',
            'Floor_Hits_Frac'
        ]
        summary = df[summary_cols].copy()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(save_path, index=False)
            print(f"Dramatic effects summary saved to {save_path}")

        return summary, df  # return df with FDR mask for heatmaps

    def create_layerwise_heatmap(self, df, focus='Early_Layers_g_trim', save_path=None, mask_non_sig=False):
        """Heatmap of effect sizes; optionally mask non-significant cells."""
        heatmap_data = df.pivot_table(
            values=focus,
            index=['Model'],
            columns=['Phenomenon', 'Metric'],
            aggfunc='first'
        )
        plt.figure(figsize=(20, 8))
        mask = heatmap_data.isnull()
        vmax = max(3.0, heatmap_data.abs().max().max()) if not heatmap_data.empty else 3.0
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            mask=mask,
            cbar_kws={'label': f"Effect Size ({focus.replace('_', ' ')})"},
            linewidths=0.5
        )
        focus_label = focus.replace('_', ' ')
        plt.title(f"Layer-Specific Effects: {focus_label}")
        plt.xlabel("Phenomenon and Metric")
        plt.ylabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer-wise heatmap ({focus}) saved to {save_path}")
        plt.close()

    def create_masked_heatmap(self, df, pcol='Early_Layers_p', focus='Early_Layers_g_trim', save_path=None, fdr_col='FDR_Significant'):
        """Masked heatmap that blanks non-significant cells (by FDR)."""
        table = df.pivot_table(values=focus, index='Model', columns=['Phenomenon','Metric'], aggfunc='first')
        sig = df.pivot_table(values=fdr_col, index='Model', columns=['Phenomenon','Metric'], aggfunc='first')
        table = table.reindex_like(sig)
        mask = ~sig.astype(bool)
        plt.figure(figsize=(20, 8))
        vmax = max(3.0, np.nanmax(np.abs(table.values))) if table.size else 3.0
        sns.heatmap(
            table, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-vmax, vmax=vmax, cbar_kws={'label': f"{focus} (masked by FDR)"}, linewidths=0.5
        )
        plt.title(f"Layer-Specific Effects (FDR-masked): {focus}")
        plt.xlabel("Phenomenon and Metric"); plt.ylabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"FDR-masked heatmap saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer-specific syntactic effect analysis (robust)")
    parser.add_argument("--pairs", nargs="+", action="append", required=True,
                        help="Pairs of JSON files: baseline comparison (can specify multiple pairs)")
    parser.add_argument("--output_dir", default="./layerwise_analysis", help="Output directory for results")
    parser.add_argument("--heatmaps", action="store_true", help="Generate layer-specific heatmaps")
    parser.add_argument("--winsor", type=float, default=0.01, help="Winsorization percentile per side")
    parser.add_argument("--trim", type=float, default=0.2, help="Trim proportion for trimmed Hedges' g")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (unused directly; kept for completeness)")
    parser.add_argument("--fdr_q", type=float, default=0.05, help="Benjamini–Hochberg FDR q")
    parser.add_argument("--mask_heatmaps", action="store_true", help="Mask heatmap cells that are not FDR-significant (early p)")
    args = parser.parse_args()

    analyzer = LayerSpecificSyntacticAnalyzer(
        winsor_p=args.winsor, trim=args.trim, alpha=args.alpha, fdr_q=args.fdr_q, robust=True, mask_heatmaps=args.mask_heatmaps
    )

    all_results = []
    for pair in args.pairs:
        if len(pair) != 2:
            print(f"Skipping invalid pair: {pair} (need exactly 2 files)")
            continue
        baseline_file, comparison_file = pair
        print(f"Analyzing layer-wise effects: {Path(baseline_file).parent.name} vs {Path(comparison_file).parent.name}")
        results_df = analyzer.analyze_phenomenon_pair_layerwise(baseline_file, comparison_file)
        if not results_df.empty:
            all_results.append(results_df)

    if not all_results:
        print("No valid pairs to analyze")
        return

    combined_df = pd.concat(all_results, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = output_dir / "layerwise_detailed_effects.csv"
    combined_df.to_csv(detailed_path, index=False)
    print(f"Detailed layer-wise results saved to {detailed_path}")

    dramatic_summary, df_with_fdr = analyzer.create_dramatic_effects_summary(
        combined_df, output_dir / "dramatic_effects_summary.csv", alpha=args.alpha, fdr_q=args.fdr_q
    )

    if args.heatmaps and len(combined_df) > 0:
        analyzer.create_layerwise_heatmap(
            combined_df, focus='Early_Layers_g_trim',
            save_path=output_dir / "early_layers_heatmap.pdf"
        )
        analyzer.create_layerwise_heatmap(
            combined_df, focus='Final_Layers_g_trim',
            save_path=output_dir / "final_layers_heatmap.pdf"
        )
        if args.mask_heatmaps:
            analyzer.create_masked_heatmap(
                df_with_fdr, pcol='Early_Layers_p', focus='Early_Layers_g_trim',
                save_path=output_dir / "early_layers_heatmap_fdr_masked.pdf"
            )
            analyzer.create_masked_heatmap(
                df_with_fdr, pcol='Final_Layers_p', focus='Final_Layers_g_trim',
                save_path=output_dir / "final_layers_heatmap_fdr_masked.pdf"
            )

    # Key findings for Qwen
    print("\nKey Findings - Layer-Specific Analysis:")
    qwen_effects = combined_df[combined_df['Model'] == 'Qwen2.5 7B']
    if len(qwen_effects) > 0:
        qwen_early_mean = qwen_effects['Early_Layers_g_trim'].abs().mean()
        qwen_max_change_mean = qwen_effects['Max_Symmetric_%_Change'].mean()
        print(f"Qwen early layers average absolute effect size (trimmed g): {qwen_early_mean:.3f}")
        print(f"Qwen average maximum symmetric % change: {qwen_max_change_mean:.1f}%")
        qwen_max_row = qwen_effects.loc[qwen_effects['Max_Symmetric_%_Change'].abs().idxmax()]
        print(f"Qwen's most dramatic change: {qwen_max_row['Max_Symmetric_%_Change']:.1f}% ")
        print(f"  in {qwen_max_row['Metric']} for {qwen_max_row['Phenomenon']}")
        print(f"  at layer {int(qwen_max_row['Max_Symmetric_%_Change_Layer'])}")

if __name__ == "__main__":
    main()
