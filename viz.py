#!/usr/bin/env python3
"""
Flexible Visualization for GSP Diagnostics
Allows multi-json inputs and figure selection (four-graph, fiedler plot, etc.)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

plt.style.use("seaborn-v0_8")

class GSPVisualizer:
    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi

        # Fixed colors/labels for canonical model ids
        self.colors = {
            "phi-3-mini": "red",
            "llama-3.2-1b": "green", 
            "qwen2.5-7b": "blue",
        }
        self.labels = {
            "phi-3-mini": "Phi-3 Mini",
            "llama-3.2-1b": "LLaMA 3.2 1B",
            "qwen2.5-7b": "Qwen2.5 7B",
        }

        # Alias substrings -> canonical ids (lowercased matching)
        self.aliases = {
            "phi-3-mini": [
                "phi-3-mini", "phi3-mini", "microsoft/phi-3-mini", "phi-3-mini-4k"
            ],
            "llama-3.2-1b": [
                "llama-3.2-1b", "llama3.2-1b", "llama32-1b", "llama-1b",
                "meta-llama/llama-3.2-1b"
            ],
            "qwen2.5-7b": [
                "qwen2.5-7b", "qwen2-7b", "qwen-7b", "qwen/qwen2.5-7b",
                "qwen2.5-7b-instruct", "qwen2-7b-instruct"
            ],
        }

        # Style cycling for multiple JSONs
        self.linestyles = ['-', '--', '-.', ':', '-', '--']
        self.alphas = [1.0, 0.8, 0.6, 0.4, 0.9, 0.7]

    def _canonicalize_model(self, key: str) -> str | None:
        k = key.strip().lower().replace("_", "-").replace(" ", "")
        # quick collapse duplicate dashes
        k = re.sub(r"-{2,}", "-", k)
        for canonical, aliases in self.aliases.items():
            for a in aliases:
                if a in k:
                    return canonical
        # If it already looks canonical, keep it; else None (ignore)
        if k in self.colors:
            return k
        return None

    def _extract_models_with_layers(self, data: dict) -> dict:
        """
        Keep only entries that look like model results (have 'layers' list).
        Normalize model keys to canonical ids.
        """
        out = {}
        for raw_key, val in data.items():
            if not isinstance(val, dict):
                continue
            if "layers" not in val or not isinstance(val["layers"], list):
                continue
            canon = self._canonicalize_model(raw_key)
            if canon is None:
                # Try to heuristically map common patterns
                # (skip unknowns quietly)
                continue
            out[canon] = val
        return out

    def load_json_models(self, path: str) -> dict:
        """Load JSON results and normalize model keys"""
        with open(path, "r") as f:
            data = json.load(f)

        alias_map = {
            "phi3mini": "phi-3-mini",
            "phi-3-mini": "phi-3-mini",
            "phi_3_mini": "phi-3-mini",
            "llama-3.2-1b": "llama-3.2-1b",
            "llama32-1b": "llama-3.2-1b",
            "llama_3_2_1b": "llama-3.2-1b",
            "qwen2.5-7b": "qwen2.5-7b",
            "qwen2-7b": "qwen2.5-7b",
            "qwen_2_5_7b": "qwen2.5-7b",
        }

        models = {}
        for k, v in data.items():
            norm_key = alias_map.get(k.lower().replace(" ", "").replace("_", "-"), k)
            models[norm_key] = v

        return models

    def _ensure_parent_dir(self, save_path: str | None):
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)

    def _extract_condition_from_filename(self, filepath: str) -> str:
        """Extract condition name from filepath for legend"""
        path = Path(filepath)
        # Get parent directory name as condition
        condition = path.parent.name
        # Clean up common patterns
        condition = condition.replace("_", " ").replace("-", " ")
        condition = condition.replace("iclr results", "").strip()
        if not condition:
            condition = path.stem
        return condition.title()

    def plot_four_graph_comparison(self, jsons, save_path=None):
        """Plot comparison across multiple JSON files"""
        if len(jsons) < 2:
            raise ValueError("Need at least 2 JSON files for comparison")

        # Load all datasets
        datasets = []
        conditions = []
        for json_path in jsons:
            data = self.load_json_models(json_path)
            datasets.append(data)
            conditions.append(self._extract_condition_from_filename(json_path))

        # Get all models across all datasets
        all_models = set()
        for dataset in datasets:
            all_models.update(dataset.keys())
        all_models = sorted(all_models)

        print(f"Plotting models: {all_models}")
        print(f"Conditions: {conditions}")

        metrics = ["energy", "hfer", "spectral_entropy", "smoothness_index"]
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for model in all_models:
                color = self.colors.get(model, "black")
                model_label = self.labels.get(model, model)
                
                # Store all values for this model to compute error bands
                model_data = []
                
                for dataset_idx, (dataset, condition) in enumerate(zip(datasets, conditions)):
                    if model not in dataset:
                        continue
                        
                    layers_data = dataset[model].get("layers", [])
                    if not layers_data:
                        continue

                    try:
                        vals = [l.get(metric, np.nan) for l in layers_data]
                        layers = [l.get("layer", i) for i, l in enumerate(layers_data)]
                    except Exception as e:
                        print(f"⚠️ Could not read {metric} for {model} in {condition}: {e}")
                        continue

                    if all(np.isnan(vals)):
                        continue

                    linestyle = self.linestyles[dataset_idx % len(self.linestyles)]
                    alpha = self.alphas[dataset_idx % len(self.alphas)]
                    
                    ax.plot(layers, vals, color=color, linewidth=2, 
                           linestyle=linestyle, alpha=alpha,
                           label=f"{model_label} - {condition}")
                    
                    model_data.append((layers, vals))
                
                # Add error band if we have multiple conditions for this model
                if len(model_data) >= 2:
                    # Find common layer range
                    min_len = min(len(data[0]) for data in model_data)
                    all_vals = np.array([data[1][:min_len] for data in model_data])
                    layers_common = model_data[0][0][:min_len]
                    
                    lower = np.nanmin(all_vals, axis=0)
                    upper = np.nanmax(all_vals, axis=0)
                    ax.fill_between(layers_common, lower, upper, color=color, alpha=0.2)

            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric.title())
            ax.legend(fontsize=7, loc='best')

        plt.tight_layout()
        self._ensure_parent_dir(save_path)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✅ Saved four-graph plot to {save_path}")
        plt.show()

    def plot_fiedler_comparison(self, jsons, save_path=None):
        """Plot Fiedler values across multiple JSON files"""
        if len(jsons) < 2:
            raise ValueError("Need at least 2 JSON files for comparison")

        # Load all datasets
        datasets = []
        conditions = []
        for json_path in jsons:
            data = self.load_json_models(json_path)
            datasets.append(data)
            conditions.append(self._extract_condition_from_filename(json_path))

        # Get all models across all datasets
        all_models = set()
        for dataset in datasets:
            all_models.update(dataset.keys())
        all_models = sorted(all_models)

        print(f"Plotting models: {all_models}")
        print(f"Conditions: {conditions}")

        plt.figure(figsize=self.figsize)
        
        for model in all_models:
            color = self.colors.get(model, "black")
            model_label = self.labels.get(model, model)
            
            model_data = []  # Store data for error bands
            
            for dataset_idx, (dataset, condition) in enumerate(zip(datasets, conditions)):
                if model not in dataset:
                    continue
                    
                layers_data = dataset[model].get("layers", [])
                if not layers_data:
                    continue

                try:
                    vals = [l.get("fiedler_value", np.nan) for l in layers_data]
                    layers = [l.get("layer", i) for i, l in enumerate(layers_data)]
                except KeyError:
                    print(f"⚠️ Missing 'fiedler_value' for {model} in {condition}; skipping.")
                    continue

                if all(np.isnan(vals)):
                    continue

                linestyle = self.linestyles[dataset_idx % len(self.linestyles)]
                alpha = self.alphas[dataset_idx % len(self.alphas)]
                
                plt.plot(layers, vals, color=color, linewidth=2, 
                        linestyle=linestyle, alpha=alpha,
                        label=f"{model_label} - {condition}")
                
                model_data.append((layers, vals))
            
            # Add error band if we have multiple conditions for this model
            if len(model_data) >= 2:
                # Find common layer range
                min_len = min(len(data[0]) for data in model_data)
                all_vals = np.array([data[1][:min_len] for data in model_data])
                layers_common = model_data[0][0][:min_len]
                
                lower = np.nanmin(all_vals, axis=0)
                upper = np.nanmax(all_vals, axis=0)
                plt.fill_between(layers_common, lower, upper, color=color, alpha=0.2)

        plt.title("Fiedler Value Comparison Across Conditions")
        plt.xlabel("Layer")
        plt.ylabel("Fiedler Value (λ₂)")
        plt.legend(fontsize=10, loc='best')
        
        self._ensure_parent_dir(save_path)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✅ Saved Fiedler comparison plot to {save_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Flexible GSP Diagnostics Visualization")
    parser.add_argument("--figure", choices=["four_graph", "fiedler_plot"], required=True,
                        help="Type of figure to generate")
    parser.add_argument("--json", nargs="+", required=True,
                        help="Path(s) to JSON files for comparison")
    parser.add_argument("--output", default=None, help="Optional output file")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--w", type=float, default=12.0, help="Figure width")
    parser.add_argument("--h", type=float, default=8.0, help="Figure height")

    args = parser.parse_args()
    viz = GSPVisualizer(figsize=(args.w, args.h), dpi=args.dpi)

    if args.figure == "four_graph":
        viz.plot_four_graph_comparison(args.json, save_path=args.output)
    elif args.figure == "fiedler_plot":
        viz.plot_fiedler_comparison(args.json, save_path=args.output)


if __name__ == "__main__":
    main()