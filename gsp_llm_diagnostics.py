#!/usr/bin/env python3
"""
Graph Signal Processing Framework for Diagnostic Explainability in Large Language Models
Implementation of the ICASSP 2026 submission

This module implements the core GSP diagnostics for transformer models including:
- Dynamic attention graph construction
- Spectral diagnostics (Dirichlet energy, spectral entropy, HFER)
- Theoretical guarantees and robustness analysis
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# HuggingFace imports
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    GPT2LMHeadModel, GPTNeoForCausalLM, LlamaForCausalLM,
    BertModel, RobertaModel
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GSPConfig:
    """Configuration for GSP diagnostics"""
    # Model parameters
    model_name: str = "gpt2"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GSP parameters
    head_aggregation: str = "uniform"  # uniform, attention_weighted, learnable
    symmetrization: str = "symmetric"  # symmetric, row_norm, col_norm
    normalization: str = "rw"  # rw (random walk), sym (symmetric), none
    hfer_cutoff_ratio: float = 0.1  # High frequency cutoff as ratio of total eigenvectors
    
    # Spectral computation
    num_eigenvalues: int = 50  # Number of eigenvalues to compute for spectral analysis
    eigen_solver: str = "sparse"  # sparse (ARPACK), dense (full eigendecomposition)
    lanczos_max_iter: int = 1000
    
    # Evaluation parameters
    batch_size: int = 1
    num_layers_analyze: Optional[int] = None  # None means all layers
    save_attention: bool = True
    save_activations: bool = True
    
    # Output parameters
    output_dir: str = "./gsp_results"
    save_plots: bool = True
    save_intermediate: bool = True
    verbose: bool = True


@dataclass
class SpectralDiagnostics:
    """Container for spectral diagnostics results"""
    layer: int
    energy: float
    smoothness_index: float
    spectral_entropy: float
    hfer: float
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray]
    spectral_masses: np.ndarray
    fiedler_value: float
    connectivity: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'layer': int(self.layer),
            'energy': float(self.energy),
            'smoothness_index': float(self.smoothness_index),
            'spectral_entropy': float(self.spectral_entropy),
            'hfer': float(self.hfer),
            'eigenvalues': self.eigenvalues.tolist(),
            'spectral_masses': self.spectral_masses.tolist(),
            'fiedler_value': float(self.fiedler_value),
            'connectivity': bool(self.connectivity)
        }


class GraphConstructor:
    """Constructs dynamic attention graphs from transformer attention patterns"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        
    def symmetrize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize attention matrix according to specified method
        Args:
            attention: [batch, heads, seq_len, seq_len] attention tensor
        Returns:
            Symmetrized attention tensor
        """
        if self.config.symmetrization == "symmetric":
            return 0.5 * (attention + attention.transpose(-2, -1))
        elif self.config.symmetrization == "row_norm":
            row_sums = attention.sum(dim=-1, keepdim=True)
            attention_norm = attention / (row_sums + 1e-8)
            return 0.5 * (attention_norm + attention_norm.transpose(-2, -1))
        elif self.config.symmetrization == "col_norm":
            col_sums = attention.sum(dim=-2, keepdim=True)
            attention_norm = attention / (col_sums + 1e-8)
            return 0.5 * (attention_norm + attention_norm.transpose(-2, -1))
        else:
            raise ValueError(f"Unknown symmetrization method: {self.config.symmetrization}")
    
    def aggregate_heads(
        self,
        attention: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # [B,Q] or broadcastable to [B,1,Q,K]
    ) -> torch.Tensor:
        """
        Aggregate multi-head attention into single adjacency matrix.
        attention:  [B, H, Q, K]
        attn_mask:  if provided, True for real tokens. Either [B,Q] or broadcastable
                    to [B,1,Q,K]. Used only for weighting in 'attention_weighted'.
        returns:    [B, Q, K]
        """
        if attention.dim() != 4:
            raise ValueError(f"Expected attention [B,H,Q,K], got {list(attention.shape)}")

        B, H, Q, K = attention.shape

        method = self.config.head_aggregation

        if method == "uniform":
            # Per-batch uniform weights over heads
            weights = attention.new_full((B, H), 1.0 / H)

        elif method == "attention_weighted":
            A = attention

            # Optional masking so padded tokens don't influence head weights
            if attn_mask is not None:
                if attn_mask.dim() == 2:           # [B,Q] -> build keep mask [B,1,Q,K]
                    qmask = attn_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,Q,1]
                    kmask = attn_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,K]
                    keep = qmask & kmask                          # [B,1,Q,K]
                else:
                    keep = attn_mask                               # already broadcastable
                A = A.masked_fill(~keep, 0.0)

            # Total “mass” per head (per batch)
            masses = A.sum(dim=(2, 3))                              # [B,H]
            weights = masses / (masses.sum(dim=1, keepdim=True) + 1e-8)

        elif method == "learnable":
            # Not implemented; fall back to uniform but keep correct shape
            logging.warning("Learnable aggregation not implemented; using uniform.")
            weights = attention.new_full((B, H), 1.0 / H)

        else:
            raise ValueError(f"Unknown head aggregation method: {method}")

        # IMPORTANT: broadcast weights on the HEAD axis only
        w = weights.view(B, H, 1, 1)                                # [B,H,1,1]
        aggregated = (attention * w).sum(dim=1)                     # [B,Q,K]
        return aggregated

    
    def construct_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Construct graph Laplacian from adjacency matrix
        Args:
            adjacency: [batch, seq_len, seq_len] adjacency matrix
        Returns:
            Graph Laplacian matrix
        """
        # Ensure non-negative weights
        adjacency = torch.clamp(adjacency, min=0)
        
        # Compute degree matrix
        degrees = adjacency.sum(dim=-1)  # [batch, seq_len]
        
        if self.config.normalization == "rw":
            # Random walk Laplacian: L = I - D^{-1}W
            deg_inv = torch.where(degrees > 1e-8, 1.0 / degrees, torch.zeros_like(degrees))
            deg_inv_diag = torch.diag_embed(deg_inv)
            laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0) - torch.matmul(deg_inv_diag, adjacency)
        elif self.config.normalization == "sym":
            # Symmetric normalized Laplacian: L = I - D^{-1/2}WD^{-1/2}
            deg_sqrt_inv = torch.where(degrees > 1e-8, 1.0 / torch.sqrt(degrees), torch.zeros_like(degrees))
            deg_sqrt_inv_diag = torch.diag_embed(deg_sqrt_inv)
            normalized_adj = torch.matmul(torch.matmul(deg_sqrt_inv_diag, adjacency), deg_sqrt_inv_diag)
            laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0) - normalized_adj
        elif self.config.normalization == "none":
            # Combinatorial Laplacian: L = D - W
            degree_diag = torch.diag_embed(degrees)
            laplacian = degree_diag - adjacency
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization}")
        
        return laplacian


class SpectralAnalyzer:
    """Performs spectral analysis and computes GSP diagnostics"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
    
    def compute_eigendecomposition(self, laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of graph Laplacian
        Args:
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            eigenvalues, eigenvectors
        """
        seq_len = laplacian.shape[0]
        
        if self.config.eigen_solver == "sparse" and seq_len > 50:
            # Use sparse eigenvalue solver for large matrices
            try:
                # Convert to sparse matrix for efficiency
                laplacian_sparse = csr_matrix(laplacian)
                
                # Compute smallest eigenvalues (including 0) and largest ones
                k_small = min(self.config.num_eigenvalues // 2, seq_len - 2)
                k_large = min(self.config.num_eigenvalues - k_small, seq_len - k_small - 1)
                
                if k_small > 0:
                    eigenvals_small, eigenvecs_small = eigsh(
                        laplacian_sparse, k=k_small, which='SM',
                        maxiter=self.config.lanczos_max_iter, tol=1e-6
                    )
                else:
                    eigenvals_small, eigenvecs_small = np.array([]), np.zeros((seq_len, 0))
                
                if k_large > 0:
                    eigenvals_large, eigenvecs_large = eigsh(
                        laplacian_sparse, k=k_large, which='LM',
                        maxiter=self.config.lanczos_max_iter, tol=1e-6
                    )
                else:
                    eigenvals_large, eigenvecs_large = np.array([]), np.zeros((seq_len, 0))
                
                # Combine and sort
                eigenvals = np.concatenate([eigenvals_small, eigenvals_large])
                eigenvecs = np.concatenate([eigenvecs_small, eigenvecs_large], axis=1)
                
                # Sort by eigenvalue
                sort_idx = np.argsort(eigenvals)
                eigenvals = eigenvals[sort_idx]
                eigenvecs = eigenvecs[:, sort_idx]
                
            except ArpackNoConvergence as e:
                logger.warning(f"ARPACK did not converge, falling back to dense solver: {e}")
                eigenvals, eigenvecs = eigh(laplacian)
        else:
            # Use dense eigenvalue solver
            eigenvals, eigenvecs = eigh(laplacian)
        
        # Ensure eigenvalues are non-negative (numerical precision)
        eigenvals = np.maximum(eigenvals, 0)
        
        return eigenvals, eigenvecs
    
    def compute_dirichlet_energy(self, signals: np.ndarray, laplacian: np.ndarray) -> float:
        """
        Compute Dirichlet energy of signals on graph
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            Total Dirichlet energy
        """
        # Energy = Tr(X^T L X)
        energy_matrix = np.dot(signals.T, np.dot(laplacian, signals))
        energy = np.trace(energy_matrix)
        return float(energy)
    
    def compute_smoothness_index(self, signals: np.ndarray, laplacian: np.ndarray) -> float:
        """
        Compute smoothness index (normalized energy)
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            Smoothness index
        """
        energy = self.compute_dirichlet_energy(signals, laplacian)
        signal_norm = np.trace(np.dot(signals.T, signals))
        
        if signal_norm < 1e-8:
            return 0.0
        
        return energy / signal_norm
    
    def compute_spectral_entropy(self, signals: np.ndarray, eigenvectors: np.ndarray) -> float:
        """
        Compute spectral entropy of signals
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            eigenvectors: [seq_len, num_eigenvectors] eigenvector matrix
        Returns:
            Spectral entropy
        """
        # Project signals onto eigenbasis
        signal_hat = np.dot(eigenvectors.T, signals)  # [num_eigenvectors, embedding_dim]
        
        # Compute spectral energies per frequency
        spectral_energies = np.sum(signal_hat**2, axis=1)  # [num_eigenvectors]
        
        # Normalize to get probability distribution
        total_energy = np.sum(spectral_energies)
        if total_energy < 1e-8:
            return 0.0
        
        spectral_probs = spectral_energies / total_energy
        
        # Compute entropy
        spectral_probs = np.maximum(spectral_probs, 1e-12)  # Avoid log(0)
        entropy = -np.sum(spectral_probs * np.log(spectral_probs))
        
        return float(entropy)
    
    def compute_hfer(self, signals: np.ndarray, eigenvectors: np.ndarray, 
                    eigenvalues: np.ndarray, cutoff_ratio: float) -> float:
        """
        Compute High-Frequency Energy Ratio
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            eigenvectors: [seq_len, num_eigenvectors] eigenvector matrix
            eigenvalues: [num_eigenvectors] eigenvalue array
            cutoff_ratio: Fraction of spectrum to consider as high-frequency
        Returns:
            HFER value
        """
        # Project signals onto eigenbasis
        signal_hat = np.dot(eigenvectors.T, signals)  # [num_eigenvectors, embedding_dim]
        
        # Compute spectral energies per frequency
        spectral_energies = np.sum(signal_hat**2, axis=1)  # [num_eigenvectors]
        
        # Determine cutoff index
        num_eigenvectors = len(eigenvalues)
        cutoff_index = int((1 - cutoff_ratio) * num_eigenvectors)
        
        # Compute HFER
        total_energy = np.sum(spectral_energies)
        if total_energy < 1e-8:
            return 0.0
        
        high_freq_energy = np.sum(spectral_energies[cutoff_index:])
        hfer = high_freq_energy / total_energy
        
        return float(hfer)
    
    def analyze_layer(self, signals: torch.Tensor, laplacian: torch.Tensor, 
                     layer_idx: int) -> SpectralDiagnostics:
        """
        Perform complete spectral analysis for a single layer
        Args:
            signals: [seq_len, embedding_dim] activation tensor
            laplacian: [seq_len, seq_len] Laplacian tensor
            layer_idx: Layer index
        Returns:
            Complete spectral diagnostics
        """
        # Convert to numpy for numerical computations
        signals_np = signals.detach().cpu().numpy()
        laplacian_np = laplacian.detach().cpu().numpy().squeeze()
        
        # Check connectivity
        connectivity = self._check_connectivity(laplacian_np)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = self.compute_eigendecomposition(laplacian_np)
        
        # Compute diagnostics
        energy = self.compute_dirichlet_energy(signals_np, laplacian_np)
        smoothness_index = self.compute_smoothness_index(signals_np, laplacian_np)
        spectral_entropy = self.compute_spectral_entropy(signals_np, eigenvectors)
        hfer = self.compute_hfer(signals_np, eigenvectors, eigenvalues, 
                               self.config.hfer_cutoff_ratio)
        
        # Compute spectral masses
        signal_hat = np.dot(eigenvectors.T, signals_np)
        spectral_masses = np.sum(signal_hat**2, axis=1)
        
        # Fiedler value (second smallest eigenvalue)
        fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        return SpectralDiagnostics(
            layer=layer_idx,
            energy=energy,
            smoothness_index=smoothness_index,
            spectral_entropy=spectral_entropy,
            hfer=hfer,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors if self.config.save_intermediate else None,
            spectral_masses=spectral_masses,
            fiedler_value=fiedler_value,
            connectivity=connectivity
        )
    
    def _check_connectivity(self, laplacian: np.ndarray) -> bool:
        """Check if the graph is connected by examining the null space of Laplacian"""
        eigenvals, _ = self.compute_eigendecomposition(laplacian)
        # Graph is connected if there's exactly one zero eigenvalue
        zero_eigenvals = np.sum(eigenvals < 1e-6)
        return zero_eigenvals == 1


class LLMInstrumenter:
    """Instruments LLM to extract attention patterns and activations"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.attention_data = {}
        self.activation_data = {}
        self.hooks = []
        
    def load_model(self, model_name: str):
        """Load HuggingFace model and tokenizer with support for custom code"""
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer with trust_remote_code if needed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=getattr(self.config, "trust_remote_code", False)
            )
        except Exception as e:
            logger.warning(f"Falling back tokenizer load for {model_name}: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare dtype and device map
        dtype = getattr(self.config, "torch_dtype", torch.float32)
        device_map = getattr(self.config, "device_map", self.config.device)

        # Try loading as causal LM first
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=getattr(self.config, "trust_remote_code", False)
            )
        except Exception as e1:
            logger.warning(f"AutoModelForCausalLM failed for {model_name}: {e1}. Trying AutoModel...")
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=getattr(self.config, "trust_remote_code", False)
                )
            except Exception as e2:
                logger.error(f"Failed to load model {model_name}: {e2}")
                raise

        self.model.eval()
        logger.info(f"Model loaded successfully. Device: {next(self.model.parameters()).device}")

    def register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        if not self.config.save_activations:
            return
        
        def create_activation_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Handle different output formats
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                self.activation_data[layer_name] = hidden_states.detach()
            return hook
        
        # Register hooks for transformer layers
        if hasattr(self.model, 'transformer'):
            # GPT-style models
            for i, layer in enumerate(self.model.transformer.h):
                hook = layer.register_forward_hook(create_activation_hook(f"layer_{i}"))
                self.hooks.append(hook)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama-style models
            for i, layer in enumerate(self.model.model.layers):
                hook = layer.register_forward_hook(create_activation_hook(f"layer_{i}"))
                self.hooks.append(hook)
        elif hasattr(self.model, 'encoder'):
            # BERT-style models
            for i, layer in enumerate(self.model.encoder.layer):
                hook = layer.register_forward_hook(create_activation_hook(f"layer_{i}"))
                self.hooks.append(hook)
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text through model and extract attention/activations
        Args:
            text: Input text string
        Returns:
            Dictionary containing processed results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Clear previous data
        self.attention_data.clear()
        self.activation_data.clear()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract attention patterns
        attentions = outputs.attentions  # Tuple of [batch, heads, seq_len, seq_len]
        hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]
        
        return {
            'inputs': inputs,
            'attentions': attentions,
            'hidden_states': hidden_states,
            'activations': dict(self.activation_data),
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'text': text
        }


class GSPDiagnosticsFramework:
    """Main framework for GSP-based LLM diagnostics"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        self.instrumenter = LLMInstrumenter(config)
        self.graph_constructor = GraphConstructor(config)
        self.spectral_analyzer = SpectralAnalyzer(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(Path(config.output_dir) / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    def analyze_text(self, text: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Perform complete GSP analysis on input text
        Args:
            text: Input text to analyze
            save_results: Whether to save intermediate results
        Returns:
            Complete analysis results
        """
        logger.info(f"Analyzing text: {text[:100]}...")
        
        # Process text through model
        model_outputs = self.instrumenter.process_text(text)
        attentions = model_outputs['attentions']
        hidden_states = model_outputs['hidden_states']
        
        # Analyze each layer
        layer_diagnostics = []
        num_layers = len(attentions)
        
        if self.config.num_layers_analyze is not None:
            num_layers = min(num_layers, self.config.num_layers_analyze)
        
        for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
            # Get layer attention and hidden states
            attention = attentions[layer_idx]  # [batch, heads, seq_len, seq_len]
            signals = hidden_states[layer_idx + 1]  # [batch, seq_len, hidden_dim]
            
            # Remove batch dimension (assuming batch_size=1)
            attention = attention.squeeze(0)  # [heads, seq_len, seq_len]
            signals = signals.squeeze(0)  # [seq_len, hidden_dim]
            
            attention_sym = self.graph_constructor.symmetrize_attention(
                attention.unsqueeze(0)
            ).squeeze(0)  # [heads, seq_len, seq_len]

            # NEW: pull the model's attention mask (if present)
            inputs_mask = model_outputs['inputs'].get('attention_mask', None)
            if inputs_mask is not None:
                inputs_mask = inputs_mask.to(torch.bool)  # [1, Q]

            adjacency = self.graph_constructor.aggregate_heads(
                attention_sym.unsqueeze(0),        # [1, H, Q, K]
                attn_mask=inputs_mask              # [1, Q] or None
            ).squeeze(0)                           # [Q, K]

            laplacian = self.graph_constructor.construct_laplacian(
                adjacency.unsqueeze(0)
            ).squeeze(0)  # [seq_len, seq_len]
            
            # Perform spectral analysis
            diagnostics = self.spectral_analyzer.analyze_layer(
                signals, laplacian, layer_idx
            )
            
            layer_diagnostics.append(diagnostics)
            
            if self.config.verbose:
                logger.info(f"Layer {layer_idx}: Energy={diagnostics.energy:.4f}, "
                          f"SMI={diagnostics.smoothness_index:.4f}, "
                          f"SE={diagnostics.spectral_entropy:.4f}, "
                          f"HFER={diagnostics.hfer:.4f}")
        
        # Compile results
        results = {
            'text': text,
            'tokens': model_outputs['tokens'],
            'layer_diagnostics': layer_diagnostics,
            'config': asdict(self.config),
            'model_outputs': model_outputs if save_results else None
        }
        
        if save_results:
            self._save_results(results)
        
        return results
    
    def analyze_dataset(self, texts: List[str], labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze multiple texts and return results as DataFrame
        Args:
            texts: List of input texts
            labels: Optional labels for texts (e.g., 'hallucination', 'factual')
        Returns:
            DataFrame with aggregated results
        """
        all_results = []
        
        for i, text in enumerate(tqdm(texts, desc="Processing dataset")):
            try:
                result = self.analyze_text(text, save_results=False)
                
                # Extract summary statistics per text
                for layer_idx, diag in enumerate(result['layer_diagnostics']):
                    row = {
                        'text_id': i,
                        'text': text,
                        'layer': layer_idx,
                        'energy': diag.energy,
                        'smoothness_index': diag.smoothness_index,
                        'spectral_entropy': diag.spectral_entropy,
                        'hfer': diag.hfer,
                        'fiedler_value': diag.fiedler_value,
                        'connectivity': diag.connectivity,
                        'num_tokens': len(result['tokens'])
                    }
                    
                    if labels is not None:
                        row['label'] = labels[i]
                    
                    all_results.append(row)
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                continue
        
        df = pd.DataFrame(all_results)
        
        # Save dataset results
        output_path = Path(self.config.output_dir) / "dataset_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset results saved to {output_path}")
        
        return df
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to disk"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_path = Path(self.config.output_dir) / f"analysis_{timestamp}.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        
        # Save diagnostics as JSON (more readable)
        diagnostics_data = []
        for diag in results['layer_diagnostics']:
            diagnostics_data.append(diag.to_dict())
        
        json_path = Path(self.config.output_dir) / f"diagnostics_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                'text': results['text'],
                'tokens': results['tokens'],
                'diagnostics': diagnostics_data
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path} and {json_path}")
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create diagnostic visualizations"""
        if not self.config.save_plots:
            return
        
        diagnostics = results['layer_diagnostics']
        num_layers = len(diagnostics)
        
        # Extract metrics for plotting
        layers = list(range(num_layers))
        energies = [d.energy for d in diagnostics]
        smoothness_indices = [d.smoothness_index for d in diagnostics]
        spectral_entropies = [d.spectral_entropy for d in diagnostics]
        hfers = [d.hfer for d in diagnostics]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"GSP Diagnostics: {results['text'][:50]}...", fontsize=14)
        
        # Energy plot
        axes[0, 0].plot(layers, energies, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Dirichlet Energy by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Smoothness Index plot
        axes[0, 1].plot(layers, smoothness_indices, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Smoothness Index by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Smoothness Index')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectral Entropy plot
        axes[1, 0].plot(layers, spectral_entropies, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Spectral Entropy by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Spectral Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HFER plot
        axes[1, 1].plot(layers, hfers, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_title('High-Frequency Energy Ratio by Layer')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('HFER')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(self.config.output_dir) / f"diagnostics_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create spectral analysis plot for selected layers
        selected_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
        selected_layers = [l for l in selected_layers if l < num_layers]
        
        fig, axes = plt.subplots(len(selected_layers), 2, figsize=(15, 4*len(selected_layers)))
        if len(selected_layers) == 1:
            axes = axes.reshape(1, -1)
        
        for i, layer_idx in enumerate(selected_layers):
            diag = diagnostics[layer_idx]
            
            # Eigenvalue spectrum
            axes[i, 0].semilogy(diag.eigenvalues, 'b-', linewidth=2)
            axes[i, 0].set_title(f'Layer {layer_idx}: Eigenvalue Spectrum')
            axes[i, 0].set_xlabel('Eigenvalue Index')
            axes[i, 0].set_ylabel('Eigenvalue (log scale)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Spectral mass distribution
            axes[i, 1].bar(range(len(diag.spectral_masses)), diag.spectral_masses, alpha=0.7)
            axes[i, 1].set_title(f'Layer {layer_idx}: Spectral Mass Distribution')
            axes[i, 1].set_xlabel('Frequency Index')
            axes[i, 1].set_ylabel('Spectral Mass')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        spectral_plot_path = Path(self.config.output_dir) / f"spectral_analysis_{timestamp}.png"
        plt.savefig(spectral_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {plot_path} and {spectral_plot_path}")
    
    def run_synthetic_validation(self) -> Dict[str, Any]:
        """
        Run synthetic validation experiments as described in the paper
        """
        logger.info("Running synthetic validation experiments...")
        
        # Generate synthetic graphs with known spectral properties
        seq_lens = [32, 64, 128]
        noise_levels = [0.1, 0.5, 1.0]
        results = {}
        
        for seq_len in seq_lens:
            for noise_level in noise_levels:
                # Create synthetic graph (path graph + noise)
                adjacency = np.zeros((seq_len, seq_len))
                # Path graph structure
                for i in range(seq_len - 1):
                    adjacency[i, i + 1] = 1
                    adjacency[i + 1, i] = 1
                
                # Add random noise
                noise = np.random.uniform(0, noise_level, (seq_len, seq_len))
                noise = 0.5 * (noise + noise.T)  # Symmetrize
                adjacency += noise
                
                # Create Laplacian
                degrees = adjacency.sum(axis=1)
                laplacian = np.diag(degrees) - adjacency
                
                # Generate synthetic signals with known frequency content
                eigenvals, eigenvecs = eigh(laplacian)
                
                # Low-frequency signal
                signal_low = eigenvecs[:, :seq_len//4] @ np.random.randn(seq_len//4, 64)
                # High-frequency signal
                signal_high = eigenvecs[:, -seq_len//4:] @ np.random.randn(seq_len//4, 64)
                # Mixed signal
                signal_mixed = 0.7 * signal_low + 0.3 * signal_high
                
                # Analyze with our framework
                signals_dict = {
                    'low_freq': signal_low,
                    'high_freq': signal_high,
                    'mixed': signal_mixed
                }
                
                for signal_type, signal in signals_dict.items():
                    # Convert to torch tensors
                    signal_torch = torch.tensor(signal, dtype=torch.float32)
                    laplacian_torch = torch.tensor(laplacian, dtype=torch.float32)
                    
                    # Analyze
                    diag = self.spectral_analyzer.analyze_layer(
                        signal_torch, laplacian_torch, 0
                    )
                    
                    key = f"seq{seq_len}_noise{noise_level}_{signal_type}"
                    results[key] = {
                        'seq_len': seq_len,
                        'noise_level': noise_level,
                        'signal_type': signal_type,
                        'energy': diag.energy,
                        'smoothness_index': diag.smoothness_index,
                        'spectral_entropy': diag.spectral_entropy,
                        'hfer': diag.hfer,
                        'expected_smoothness': 'high' if signal_type == 'low_freq' else 'low'
                    }
        
        # Save synthetic results
        synthetic_path = Path(self.config.output_dir) / "synthetic_validation.json"
        with open(synthetic_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Synthetic validation completed. Results saved to {synthetic_path}")
        return results
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.instrumenter.cleanup_hooks()


def create_benchmark_dataset() -> Tuple[List[str], List[str]]:
    """
    Create a benchmark dataset for testing hallucination detection
    Returns:
        texts, labels (where labels are 'factual' or 'hallucination')
    """
    # Factual statements
    factual_texts = [
        "The capital of France is Paris, which is located in northern France.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The Earth orbits around the Sun once every 365.25 days approximately.",
        "Shakespeare wrote Romeo and Juliet in the late 16th century.",
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "Python is a high-level programming language created by Guido van Rossum.",
        "The Great Wall of China was built over many centuries to protect against invasions.",
        "DNA stands for deoxyribonucleic acid and contains genetic information.",
        "The periodic table organizes chemical elements by their atomic number.",
        "Albert Einstein developed the theory of relativity in the early 20th century."
    ]
    
    # Potentially hallucinated/false statements
    hallucination_texts = [
        "The capital of France is London, which is famous for its Eiffel Tower.",
        "Water boils at 50 degrees Celsius at standard atmospheric pressure.",
        "The Earth orbits around the Moon once every 200 days approximately.",
        "Shakespeare wrote Romeo and Juliet in the 19th century during the Victorian era.",
        "The speed of light in vacuum is approximately 150,000,000 meters per second.",
        "Python is a low-level programming language created by Bill Gates in the 1970s.",
        "The Great Wall of China was built in the 21st century using modern concrete.",
        "DNA stands for dynamic nucleic acid and controls human emotions directly.",
        "The periodic table organizes chemical elements by their color and taste.",
        "Albert Einstein developed the theory of quantum mechanics in the 18th century."
    ]
    
    texts = factual_texts + hallucination_texts
    labels = ['factual'] * len(factual_texts) + ['hallucination'] * len(hallucination_texts)
    
    return texts, labels


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Graph Signal Processing Framework for LLM Diagnostics"
    )
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="gpt2",
                      help="HuggingFace model name")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use (cuda/cpu/auto)")
    
    # GSP parameters
    parser.add_argument("--head_aggregation", type=str, default="uniform",
                      choices=["uniform", "attention_weighted", "learnable"],
                      help="Method to aggregate attention heads")
    parser.add_argument("--symmetrization", type=str, default="symmetric",
                      choices=["symmetric", "row_norm", "col_norm"],
                      help="Method to symmetrize attention matrices")
    parser.add_argument("--normalization", type=str, default="rw",
                      choices=["rw", "sym", "none"],
                      help="Laplacian normalization method")
    parser.add_argument("--hfer_cutoff_ratio", type=float, default=0.1,
                      help="High-frequency cutoff ratio")
    
    # Spectral computation
    parser.add_argument("--num_eigenvalues", type=int, default=50,
                      help="Number of eigenvalues to compute")
    parser.add_argument("--eigen_solver", type=str, default="sparse",
                      choices=["sparse", "dense"],
                      help="Eigenvalue solver method")
    
    # Evaluation parameters
    parser.add_argument("--num_layers_analyze", type=int, default=None,
                      help="Number of layers to analyze (None for all)")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for processing")
    
    # Input/Output
    parser.add_argument("--text", type=str, default=None,
                      help="Single text to analyze")
    parser.add_argument("--input_file", type=str, default=None,
                      help="File containing texts to analyze (one per line)")
    parser.add_argument("--output_dir", type=str, default="./gsp_results",
                      help="Output directory for results")
    parser.add_argument("--save_plots", action="store_true",
                      help="Save diagnostic plots")
    parser.add_argument("--save_intermediate", action="store_true",
                      help="Save intermediate computation results")
    parser.add_argument("--verbose", action="store_true",
                      help="Verbose output")
    
    # Experiments
    parser.add_argument("--run_synthetic", action="store_true",
                      help="Run synthetic validation experiments")
    parser.add_argument("--run_benchmark", action="store_true",
                      help="Run benchmark dataset evaluation")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Create configuration
    config = GSPConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        device=device,
        head_aggregation=args.head_aggregation,
        symmetrization=args.symmetrization,
        normalization=args.normalization,
        hfer_cutoff_ratio=args.hfer_cutoff_ratio,
        num_eigenvalues=args.num_eigenvalues,
        eigen_solver=args.eigen_solver,
        num_layers_analyze=args.num_layers_analyze,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        save_intermediate=args.save_intermediate,
        verbose=args.verbose
    )
    
    # Initialize framework
    with GSPDiagnosticsFramework(config) as framework:
        # Load model
        framework.instrumenter.load_model(args.model_name)
        framework.instrumenter.register_hooks()
        
        # Run synthetic validation if requested
        if args.run_synthetic:
            framework.run_synthetic_validation()
        
        # Run benchmark evaluation if requested
        if args.run_benchmark:
            logger.info("Running benchmark evaluation...")
            texts, labels = create_benchmark_dataset()
            df = framework.analyze_dataset(texts, labels)
            
            # Generate summary statistics
            summary = df.groupby(['label', 'layer']).agg({
                'energy': ['mean', 'std'],
                'smoothness_index': ['mean', 'std'],
                'spectral_entropy': ['mean', 'std'],
                'hfer': ['mean', 'std']
            }).round(4)
            
            print("\nBenchmark Results Summary:")
            print(summary)
            
            # Save summary
            summary_path = Path(args.output_dir) / "benchmark_summary.csv"
            summary.to_csv(summary_path)
            logger.info(f"Benchmark summary saved to {summary_path}")
        
        # Process single text if provided
        if args.text:
            logger.info(f"Analyzing single text: {args.text}")
            results = framework.analyze_text(args.text)
            framework.create_visualizations(results)
            
            # Print summary
            print(f"\nAnalysis Results for: '{args.text[:50]}...'")
            print("-" * 60)
            for i, diag in enumerate(results['layer_diagnostics']):
                print(f"Layer {i:2d}: Energy={diag.energy:8.4f}, "
                      f"SMI={diag.smoothness_index:6.4f}, "
                      f"SE={diag.spectral_entropy:6.4f}, "
                      f"HFER={diag.hfer:6.4f}")
        
        # Process input file if provided
        if args.input_file:
            logger.info(f"Processing input file: {args.input_file}")
            with open(args.input_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            df = framework.analyze_dataset(texts)
            
            # Generate summary statistics by text
            text_summary = df.groupby('text_id').agg({
                'energy': 'mean',
                'smoothness_index': 'mean',
                'spectral_entropy': 'mean',
                'hfer': 'mean'
            }).round(4)
            
            print("\nFile Processing Results Summary:")
            print(text_summary)
        
        # If no specific task, show usage
        if not any([args.text, args.input_file, args.run_synthetic, args.run_benchmark]):
            print("No analysis task specified. Use --help for usage information.")
            print("Example: python gsp_llm_diagnostics.py --text 'Hello world' --save_plots")


if __name__ == "__main__":
    main()