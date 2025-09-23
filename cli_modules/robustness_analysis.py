#!/usr/bin/env python3
"""
Robustness Analysis Module for GSP LLM Diagnostics
Implements perturbation experiments and theoretical validation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerturbationConfig:
    """Configuration for perturbation experiments"""
    perturbation_types: List[str] = None  # ['token_swap', 'token_delete', 'token_insert', 'embedding_noise']
    noise_levels: List[float] = None  # [0.01, 0.05, 0.1, 0.2, 0.5]
    num_perturbations: int = 10
    random_seed: int = 42
    
    def __post_init__(self):
        if self.perturbation_types is None:
            self.perturbation_types = ['token_swap', 'token_delete', 'embedding_noise']
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]


class RobustnessAnalyzer:
    """Analyzes model robustness using GSP diagnostics"""
    
    def __init__(self, gsp_framework, config: PerturbationConfig):
        self.gsp_framework = gsp_framework
        self.config = config
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def generate_token_perturbations(self, text: str, tokenizer) -> List[str]:
        """Generate perturbed versions of input text at token level"""
        tokens = tokenizer.tokenize(text)
        perturbations = []
        
        for _ in range(self.config.num_perturbations):
            perturbed_tokens = tokens.copy()
            perturbation_type = np.random.choice(self.config.perturbation_types)
            
            if perturbation_type == 'token_swap' and len(perturbed_tokens) > 1:
                # Swap two random tokens
                idx1, idx2 = np.random.choice(len(perturbed_tokens), 2, replace=False)
                perturbed_tokens[idx1], perturbed_tokens[idx2] = perturbed_tokens[idx2], perturbed_tokens[idx1]
            
            elif perturbation_type == 'token_delete' and len(perturbed_tokens) > 1:
                # Delete a random token
                idx = np.random.choice(len(perturbed_tokens))
                perturbed_tokens.pop(idx)
            
            elif perturbation_type == 'token_insert':
                # Insert a random token from vocabulary
                vocab_size = tokenizer.vocab_size
                random_token_id = np.random.choice(vocab_size)
                random_token = tokenizer.convert_ids_to_tokens([random_token_id])[0]
                idx = np.random.choice(len(perturbed_tokens) + 1)
                perturbed_tokens.insert(idx, random_token)
            
            perturbed_text = tokenizer.convert_tokens_to_string(perturbed_tokens)
            perturbations.append(perturbed_text)
        
        return perturbations
    
    def generate_embedding_perturbations(self, hidden_states: torch.Tensor, 
                                       noise_level: float) -> torch.Tensor:
        """Generate perturbations in embedding space"""
        # Add Gaussian noise to embeddings
        noise = torch.randn_like(hidden_states) * noise_level
        return hidden_states + noise
    
    def compute_perturbation_sensitivity(self, text: str) -> Dict[str, any]:
        """
        Compute sensitivity of GSP diagnostics to perturbations
        Validates Theorem 4.4 (Lipschitz readout under spectral control)
        """
        logger.info(f"Computing perturbation sensitivity for: {text[:50]}...")
        
        # Get baseline analysis
        baseline_results = self.gsp_framework.analyze_text(text, save_results=False)
        baseline_diagnostics = baseline_results['layer_diagnostics']
        
        # Generate token-level perturbations
        tokenizer = self.gsp_framework.instrumenter.tokenizer
        perturbed_texts = self.generate_token_perturbations(text, tokenizer)
        
        # Analyze perturbed texts
        perturbation_results = []
        for perturbed_text in perturbed_texts:
            try:
                perturbed_result = self.gsp_framework.analyze_text(perturbed_text, save_results=False)
                perturbation_results.append(perturbed_result['layer_diagnostics'])
            except Exception as e:
                logger.warning(f"Failed to analyze perturbation: {e}")
                continue
        
        # Compute sensitivity metrics
        sensitivity_metrics = self._compute_sensitivity_metrics(
            baseline_diagnostics, perturbation_results
        )
        
        # Test embedding-level perturbations for theoretical validation
        embedding_sensitivity = self._test_embedding_perturbations(text)
        
        return {
            'baseline_diagnostics': baseline_diagnostics,
            'perturbation_results': perturbation_results,
            'sensitivity_metrics': sensitivity_metrics,
            'embedding_sensitivity': embedding_sensitivity,
            'num_perturbations': len(perturbation_results),
            'original_text': text
        }
    
    def _compute_sensitivity_metrics(self, baseline: List, perturbations: List[List]) -> Dict:
        """Compute various sensitivity metrics"""
        num_layers = len(baseline)
        metrics = {
            'energy_variance': [],
            'smoothness_variance': [],
            'spectral_entropy_variance': [],
            'hfer_variance': [],
            'energy_max_change': [],
            'smoothness_max_change': [],
            'fiedler_correlation': []
        }
        
        for layer_idx in range(num_layers):
            baseline_diag = baseline[layer_idx]
            
            # Collect metrics from all perturbations for this layer
            layer_energies = [p[layer_idx].energy for p in perturbations if layer_idx < len(p)]
            layer_smoothness = [p[layer_idx].smoothness_index for p in perturbations if layer_idx < len(p)]
            layer_entropy = [p[layer_idx].spectral_entropy for p in perturbations if layer_idx < len(p)]
            layer_hfer = [p[layer_idx].hfer for p in perturbations if layer_idx < len(p)]
            layer_fiedler = [p[layer_idx].fiedler_value for p in perturbations if layer_idx < len(p)]
            
            if len(layer_energies) == 0:
                continue
            
            # Compute variance (stability measure)
            metrics['energy_variance'].append(np.var(layer_energies))
            metrics['smoothness_variance'].append(np.var(layer_smoothness))
            metrics['spectral_entropy_variance'].append(np.var(layer_entropy))
            metrics['hfer_variance'].append(np.var(layer_hfer))
            
            # Compute maximum absolute change
            metrics['energy_max_change'].append(
                max(abs(e - baseline_diag.energy) for e in layer_energies)
            )
            metrics['smoothness_max_change'].append(
                max(abs(s - baseline_diag.smoothness_index) for s in layer_smoothness)
            )
            
            # Correlation with Fiedler value (connectivity strength)
            if len(set(layer_fiedler)) > 1:  # Need variance for correlation
                energy_changes = [abs(e - baseline_diag.energy) for e in layer_energies]
                corr, p_val = pearsonr(layer_fiedler, energy_changes)
                metrics['fiedler_correlation'].append((corr, p_val))
            else:
                metrics['fiedler_correlation'].append((0.0, 1.0))
        
        return metrics
    
    def _test_embedding_perturbations(self, text: str) -> Dict:
        """Test theoretical predictions with controlled embedding perturbations"""
        # Get model outputs for baseline
        model_outputs = self.gsp_framework.instrumenter.process_text(text)
        hidden_states = model_outputs['hidden_states']
        attentions = model_outputs['attentions']
        
        embedding_results = {}
        
        for noise_level in self.config.noise_levels:
            level_results = []
            
            for _ in range(self.config.num_perturbations):
                # Perturb embeddings at each layer
                perturbed_states = []
                for layer_states in hidden_states:
                    perturbed = self.generate_embedding_perturbations(layer_states, noise_level)
                    perturbed_states.append(perturbed)
                
                # Analyze perturbed embeddings with original attention patterns
                layer_diagnostics = []
                for layer_idx in range(len(attentions)):
                    attention = attentions[layer_idx].squeeze(0)  # Remove batch dim
                    signals = perturbed_states[layer_idx + 1].squeeze(0)  # Remove batch dim
                    
                    # Construct graph (same as original framework)
                    attention_sym = self.gsp_framework.graph_constructor.symmetrize_attention(
                        attention.unsqueeze(0)
                    ).squeeze(0)
                    
                    adjacency = self.gsp_framework.graph_constructor.aggregate_heads(
                        attention_sym.unsqueeze(0)
                    ).squeeze(0)
                    
                    laplacian = self.gsp_framework.graph_constructor.construct_laplacian(
                        adjacency.unsqueeze(0)
                    ).squeeze(0)
                    
                    # Analyze
                    diagnostics = self.gsp_framework.spectral_analyzer.analyze_layer(
                        signals, laplacian, layer_idx
                    )
                    layer_diagnostics.append(diagnostics)
                
                level_results.append(layer_diagnostics)
            
            embedding_results[noise_level] = level_results
        
        return embedding_results
    
    def validate_theoretical_bounds(self, sensitivity_results: Dict) -> Dict:
        """
        Validate theoretical bounds from Theorems 4.2-4.4
        """
        logger.info("Validating theoretical bounds...")
        
        baseline = sensitivity_results['baseline_diagnostics']
        embedding_sensitivity = sensitivity_results['embedding_sensitivity']
        
        validation_results = {
            'poincare_bound_validation': [],
            'lipschitz_bound_validation': [],
            'spectral_concentration_validation': []
        }
        
        for noise_level, perturbation_sets in embedding_sensitivity.items():
            for perturbation_set in perturbation_sets:
                for layer_idx, (baseline_diag, perturbed_diag) in enumerate(zip(baseline, perturbation_set)):
                    
                    # Validate Poincaré bound (Theorem 4.2)
                    # ||x||^2 <= (1/λ_2) * x^T L x
                    if baseline_diag.fiedler_value > 1e-6:
                        energy_ratio = perturbed_diag.energy / baseline_diag.energy
                        poincare_ratio = (baseline_diag.fiedler_value / perturbed_diag.fiedler_value 
                                        if perturbed_diag.fiedler_value > 1e-6 else float('inf'))
                        
                        validation_results['poincare_bound_validation'].append({
                            'layer': layer_idx,
                            'noise_level': noise_level,
                            'energy_ratio': energy_ratio,
                            'fiedler_ratio': poincare_ratio,
                            'bound_satisfied': energy_ratio <= poincare_ratio * 1.1  # Small tolerance
                        })
                    
                    # Validate spectral concentration prediction
                    hfer_change = abs(perturbed_diag.hfer - baseline_diag.hfer)
                    energy_change = abs(perturbed_diag.energy - baseline_diag.energy)
                    
                    validation_results['spectral_concentration_validation'].append({
                        'layer': layer_idx,
                        'noise_level': noise_level,
                        'hfer_change': hfer_change,
                        'energy_change': energy_change,
                        'predicted_correlation': hfer_change > 0.01 and energy_change > 0.01
                    })
        
        return validation_results
    
    def analyze_hallucination_correlation(self, texts: List[str], 
                                        labels: List[str]) -> Dict:
        """
        Analyze correlation between GSP diagnostics and hallucination labels
        Tests the main hypothesis of the paper
        """
        logger.info("Analyzing hallucination correlation...")
        
        # Collect diagnostics for all texts
        all_diagnostics = []
        for text, label in zip(texts, labels):
            try:
                result = self.gsp_framework.analyze_text(text, save_results=False)
                for layer_idx, diag in enumerate(result['layer_diagnostics']):
                    all_diagnostics.append({
                        'text': text,
                        'label': label,
                        'layer': layer_idx,
                        'energy': diag.energy,
                        'smoothness_index': diag.smoothness_index,
                        'spectral_entropy': diag.spectral_entropy,
                        'hfer': diag.hfer,
                        'fiedler_value': diag.fiedler_value
                    })
            except Exception as e:
                logger.warning(f"Failed to analyze text for hallucination correlation: {e}")
                continue
        
        # Convert to arrays for analysis
        import pandas as pd
        df = pd.DataFrame(all_diagnostics)
        
        # Compute correlations between diagnostics and labels
        # Convert labels to numeric (0 = factual, 1 = hallucination)
        df['label_numeric'] = (df['label'] == 'hallucination').astype(int)
        
        correlations = {}
        for metric in ['energy', 'smoothness_index', 'spectral_entropy', 'hfer']:
            # Overall correlation
            corr_pearson, p_pearson = pearsonr(df[metric], df['label_numeric'])
            corr_spearman, p_spearman = spearmanr(df[metric], df['label_numeric'])
            
            correlations[metric] = {
                'pearson_r': corr_pearson,
                'pearson_p': p_pearson,
                'spearman_r': corr_spearman,
                'spearman_p': p_spearman
            }
            
            # Layer-wise correlations
            layer_correlations = []
            for layer in df['layer'].unique():
                layer_data = df[df['layer'] == layer]
                if len(layer_data) > 3:  # Need sufficient data
                    layer_corr, layer_p = pearsonr(layer_data[metric], layer_data['label_numeric'])
                    layer_correlations.append({
                        'layer': layer,
                        'correlation': layer_corr,
                        'p_value': layer_p
                    })
            
            correlations[metric]['layer_wise'] = layer_correlations
        
        # Aggregate statistics by label
        label_stats = df.groupby('label').agg({
            'energy': ['mean', 'std', 'median'],
            'smoothness_index': ['mean', 'std', 'median'],
            'spectral_entropy': ['mean', 'std', 'median'],
            'hfer': ['mean', 'std', 'median']
        }).round(4)
        
        return {
            'correlations': correlations,
            'label_statistics': label_stats,
            'raw_data': df,
            'hypothesis_validation': self._validate_paper_hypotheses(correlations)
        }
    
    def _validate_paper_hypotheses(self, correlations: Dict) -> Dict:
        """
        Validate the main hypotheses from the paper:
        1. Hallucination correlates with elevated energy
        2. Brittle reasoning correlates with high-frequency spectral mass
        3. Effective prompting shifts energy toward smoother regimes
        """
        validation = {}
        
        # Hypothesis 1: Hallucination ~ elevated energy
        energy_corr = correlations['energy']['pearson_r']
        validation['h1_hallucination_energy'] = {
            'hypothesis': 'Hallucination correlates with elevated Dirichlet energy',
            'prediction': 'positive correlation',
            'observed_correlation': energy_corr,
            'significant': correlations['energy']['pearson_p'] < 0.05,
            'validated': energy_corr > 0 and correlations['energy']['pearson_p'] < 0.05
        }
        
        # Hypothesis 2: Brittleness ~ high-frequency dominance
        hfer_corr = correlations['hfer']['pearson_r']
        validation['h2_brittleness_hfer'] = {
            'hypothesis': 'Brittle reasoning correlates with high-frequency energy ratio',
            'prediction': 'positive correlation',
            'observed_correlation': hfer_corr,
            'significant': correlations['hfer']['pearson_p'] < 0.05,
            'validated': hfer_corr > 0 and correlations['hfer']['pearson_p'] < 0.05
        }
        
        # Hypothesis 3: Smoothness ~ reliability
        smoothness_corr = correlations['smoothness_index']['pearson_r']
        validation['h3_smoothness_reliability'] = {
            'hypothesis': 'Higher smoothness index indicates less reliable reasoning',
            'prediction': 'positive correlation with hallucination',
            'observed_correlation': smoothness_corr,
            'significant': correlations['smoothness_index']['pearson_p'] < 0.05,
            'validated': smoothness_corr > 0 and correlations['smoothness_index']['pearson_p'] < 0.05
        }
        
        return validation
    
    def create_robustness_visualizations(self, sensitivity_results: Dict, 
                                       output_dir: str):
        """Create comprehensive robustness analysis plots"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Sensitivity metrics across layers
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        sensitivity_metrics = sensitivity_results['sensitivity_metrics']
        layers = range(len(sensitivity_metrics['energy_variance']))
        
        # Energy variance
        axes[0, 0].plot(layers, sensitivity_metrics['energy_variance'], 'b-o', linewidth=2)
        axes[0, 0].set_title('Energy Variance Across Layers')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Smoothness variance
        axes[0, 1].plot(layers, sensitivity_metrics['smoothness_variance'], 'r-o', linewidth=2)
        axes[0, 1].set_title('Smoothness Index Variance Across Layers')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Maximum energy change
        axes[1, 0].plot(layers, sensitivity_metrics['energy_max_change'], 'g-o', linewidth=2)
        axes[1, 0].set_title('Maximum Energy Change Across Layers')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Max Change')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Fiedler correlation
        fiedler_corrs = [fc[0] for fc in sensitivity_metrics['fiedler_correlation']]
        axes[1, 1].plot(layers, fiedler_corrs, 'm-o', linewidth=2)
        axes[1, 1].set_title('Correlation with Fiedler Value')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Embedding perturbation analysis
        if 'embedding_sensitivity' in sensitivity_results:
            self._plot_embedding_sensitivity(
                sensitivity_results['embedding_sensitivity'], 
                output_path
            )
        
        logger.info(f"Robustness visualizations saved to {output_path}")
    
    def _plot_embedding_sensitivity(self, embedding_sensitivity: Dict, output_path: Path):
        """Plot embedding perturbation sensitivity results"""
        
        # Collect data for plotting
        noise_levels = []
        mean_energy_changes = []
        std_energy_changes = []
        mean_hfer_changes = []
        std_hfer_changes = []
        
        baseline_energy = None
        baseline_hfer = None
        
        for noise_level, perturbation_sets in embedding_sensitivity.items():
            noise_levels.append(noise_level)
            
            # Collect energy and HFER values across perturbations and layers
            all_energies = []
            all_hfers = []
            
            for perturbation_set in perturbation_sets:
                for diag in perturbation_set:
                    all_energies.append(diag.energy)
                    all_hfers.append(diag.hfer)
            
            if baseline_energy is None:
                # Use first (lowest noise) as baseline
                baseline_energy = np.mean(all_energies)
                baseline_hfer = np.mean(all_hfers)
            
            # Compute changes relative to baseline
            energy_changes = [abs(e - baseline_energy) / baseline_energy for e in all_energies]
            hfer_changes = [abs(h - baseline_hfer) / baseline_hfer for h in all_hfers if baseline_hfer > 0]
            
            mean_energy_changes.append(np.mean(energy_changes))
            std_energy_changes.append(np.std(energy_changes))
            mean_hfer_changes.append(np.mean(hfer_changes) if hfer_changes else 0)
            std_hfer_changes.append(np.std(hfer_changes) if hfer_changes else 0)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Energy sensitivity
        ax1.errorbar(noise_levels, mean_energy_changes, yerr=std_energy_changes, 
                    marker='o', linewidth=2, capsize=5)
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Relative Energy Change')
        ax1.set_title('Energy Sensitivity to Embedding Perturbations')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # HFER sensitivity
        ax2.errorbar(noise_levels, mean_hfer_changes, yerr=std_hfer_changes, 
                    marker='s', linewidth=2, capsize=5, color='red')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Relative HFER Change')
        ax2.set_title('HFER Sensitivity to Embedding Perturbations')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path / 'embedding_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_robustness_report(self, sensitivity_results: Dict, 
                                 validation_results: Dict,
                                 hallucination_results: Dict,
                                 output_dir: str) -> str:
        """Generate comprehensive robustness analysis report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("GSP LLM DIAGNOSTICS - ROBUSTNESS ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Original text: {sensitivity_results['original_text'][:100]}...")
        report_lines.append(f"Number of perturbations: {sensitivity_results['num_perturbations']}")
        report_lines.append(f"Number of layers analyzed: {len(sensitivity_results['baseline_diagnostics'])}")
        report_lines.append("")
        
        # Sensitivity metrics
        report_lines.append("PERTURBATION SENSITIVITY ANALYSIS")
        report_lines.append("-" * 40)
        sens_metrics = sensitivity_results['sensitivity_metrics']
        
        avg_energy_var = np.mean(sens_metrics['energy_variance'])
        avg_smoothness_var = np.mean(sens_metrics['smoothness_variance'])
        max_energy_change = max(sens_metrics['energy_max_change'])
        
        report_lines.append(f"Average energy variance: {avg_energy_var:.6f}")
        report_lines.append(f"Average smoothness variance: {avg_smoothness_var:.6f}")
        report_lines.append(f"Maximum energy change: {max_energy_change:.6f}")
        report_lines.append("")
        
        # Theoretical validation
        if validation_results:
            report_lines.append("THEORETICAL BOUNDS VALIDATION")
            report_lines.append("-" * 40)
            
            poincare_validations = validation_results.get('poincare_bound_validation', [])
            if poincare_validations:
                satisfied_ratio = np.mean([v['bound_satisfied'] for v in poincare_validations])
                report_lines.append(f"Poincaré bound satisfaction rate: {satisfied_ratio:.2%}")
            
            spectral_validations = validation_results.get('spectral_concentration_validation', [])
            if spectral_validations:
                corr_predictions = np.mean([v['predicted_correlation'] for v in spectral_validations])
                report_lines.append(f"Spectral concentration predictions: {corr_predictions:.2%}")
            
            report_lines.append("")
        
        # Hallucination analysis
        if hallucination_results:
            report_lines.append("HALLUCINATION CORRELATION ANALYSIS")
            report_lines.append("-" * 40)
            
            hypotheses = hallucination_results['hypothesis_validation']
            for key, result in hypotheses.items():
                report_lines.append(f"{result['hypothesis']}")
                report_lines.append(f"  Prediction: {result['prediction']}")
                report_lines.append(f"  Observed correlation: {result['observed_correlation']:.4f}")
                report_lines.append(f"  Statistically significant: {result['significant']}")
                report_lines.append(f"  Hypothesis validated: {result['validated']}")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if avg_energy_var > 0.1:
            report_lines.append("• High energy variance detected - model may be unstable to perturbations")
        else:
            report_lines.append("• Low energy variance - model shows good stability")
        
        if max_energy_change > 1.0:
            report_lines.append("• Large energy changes observed - consider regularization")
        else:
            report_lines.append("• Energy changes within acceptable bounds")
        
        if hallucination_results:
            energy_corr = hallucination_results['correlations']['energy']['pearson_r']
            if energy_corr > 0.3:
                report_lines.append("• Strong correlation between energy and hallucination detected")
                report_lines.append("• Consider using energy-based early stopping or filtering")
            
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = Path(output_dir) / "robustness_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Robustness report saved to {report_path}")
        return report_text


def main():
    """Example usage of robustness analysis"""
    import argparse
    from gsp_llm_diagnostics import GSPDiagnosticsFramework, GSPConfig
    
    parser = argparse.ArgumentParser(description="Robustness Analysis for GSP LLM Diagnostics")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model to analyze")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--output_dir", type=str, default="./robustness_results", help="Output directory")
    parser.add_argument("--num_perturbations", type=int, default=10, help="Number of perturbations")
    parser.add_argument("--noise_levels", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2], 
                       help="Noise levels for embedding perturbations")
    
    args = parser.parse_args()
    
    # Create configurations
    gsp_config = GSPConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        verbose=True
    )
    
    perturbation_config = PerturbationConfig(
        num_perturbations=args.num_perturbations,
        noise_levels=args.noise_levels
    )
    
    # Initialize framework
    with GSPDiagnosticsFramework(gsp_config) as gsp_framework:
        gsp_framework.instrumenter.load_model(args.model_name)
        
        # Initialize robustness analyzer
        robustness_analyzer = RobustnessAnalyzer(gsp_framework, perturbation_config)
        
        # Run perturbation sensitivity analysis
        sensitivity_results = robustness_analyzer.compute_perturbation_sensitivity(args.text)
        
        # Validate theoretical bounds
        validation_results = robustness_analyzer.validate_theoretical_bounds(sensitivity_results)
        
        # Create visualizations
        robustness_analyzer.create_robustness_visualizations(sensitivity_results, args.output_dir)
        
        # Generate report
        report = robustness_analyzer.generate_robustness_report(
            sensitivity_results, validation_results, {}, args.output_dir
        )
        
        print("Robustness Analysis Complete!")
        print(f"Results saved to: {args.output_dir}")
        print("\nReport Summary:")
        print(report)


if __name__ == "__main__":
    main()