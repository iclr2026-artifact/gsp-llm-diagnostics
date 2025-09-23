#!/usr/bin/env python3
"""
Evaluation and Benchmarking Module for GSP LLM Diagnostics
Implements comprehensive evaluation protocols as described in the paper
"""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments"""
    # Dataset settings
    benchmark_datasets: List[str] = None  # ['factual_qa', 'hallucination_detection', 'reasoning_tasks']
    synthetic_experiments: bool = True
    real_world_experiments: bool = True
    
    # Evaluation protocols
    cross_validation_folds: int = 5
    test_split_ratio: float = 0.2
    random_seed: int = 42
    
    # Performance metrics
    classification_metrics: List[str] = None
    regression_metrics: List[str] = None
    
    # Computational analysis
    measure_runtime: bool = True
    measure_memory: bool = True
    scalability_analysis: bool = True
    
    def __post_init__(self):
        if self.benchmark_datasets is None:
            self.benchmark_datasets = ['factual_qa', 'hallucination_detection', 'reasoning_tasks']
        if self.classification_metrics is None:
            self.classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        if self.regression_metrics is None:
            self.regression_metrics = ['mse', 'mae', 'r2']


class DatasetGenerator:
    """Generates synthetic and real-world datasets for evaluation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def create_factual_qa_dataset(self, size: int = 100) -> Tuple[List[str], List[str]]:
        """Create factual QA dataset with known correct/incorrect answers"""
        
        # Factual Q&A pairs
        factual_pairs = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
            ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
            ("In what year did World War II end?", "World War II ended in 1945."),
            ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
            ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
            ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second."),
            ("What is the currency of Japan?", "The currency of Japan is the yen."),
            ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
            ("What is the boiling point of water at sea level?", "Water boils at 100 degrees Celsius at sea level.")
        ]
        
        # Hallucinated/incorrect answers
        hallucinated_pairs = [
            ("What is the capital of France?", "The capital of France is London."),
            ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by Charles Dickens."),
            ("What is the chemical symbol for water?", "The chemical symbol for water is CO2."),
            ("In what year did World War II end?", "World War II ended in 1965."),
            ("What is the largest planet in our solar system?", "Earth is the largest planet in our solar system."),
            ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Vincent van Gogh."),
            ("What is the speed of light?", "The speed of light is approximately 150,000,000 meters per second."),
            ("What is the currency of Japan?", "The currency of Japan is the dollar."),
            ("Who discovered penicillin?", "Penicillin was discovered by Marie Curie."),
            ("What is the boiling point of water at sea level?", "Water boils at 200 degrees Celsius at sea level.")
        ]
        
        # Generate dataset
        texts = []
        labels = []
        
        num_factual = size // 2
        num_hallucinated = size - num_factual
        
        # Sample factual examples
        for i in range(num_factual):
            q, a = factual_pairs[i % len(factual_pairs)]
            text = f"Question: {q} Answer: {a}"
            texts.append(text)
            labels.append('factual')
        
        # Sample hallucinated examples
        for i in range(num_hallucinated):
            q, a = hallucinated_pairs[i % len(hallucinated_pairs)]
            text = f"Question: {q} Answer: {a}"
            texts.append(text)
            labels.append('hallucination')
        
        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    def create_reasoning_dataset(self, size: int = 100) -> Tuple[List[str], List[str]]:
        """Create multi-step reasoning dataset"""
        
        reasoning_templates = [
            # Arithmetic reasoning
            {
                'template': "If a store sells {item} for ${price} each and you buy {quantity}, how much do you spend?",
                'correct': "You spend ${item} × {quantity} = ${total}.",
                'incorrect': "You spend ${item} + {quantity} = ${wrong_total}."
            },
            # Logical reasoning
            {
                'template': "All {category1} are {property}. {item} is a {category1}. Therefore, {item} is what?",
                'correct': "Therefore, {item} is {property}.",
                'incorrect': "Therefore, {item} is not {property}."
            },
            # Temporal reasoning
            {
                'template': "If it's {time1} now and the meeting is in {duration} hours, what time is the meeting?",
                'correct': "The meeting is at {correct_time}.",
                'incorrect': "The meeting is at {incorrect_time}."
            }
        ]
        
        texts = []
        labels = []
        
        for i in range(size):
            template = reasoning_templates[i % len(reasoning_templates)]
            
            # Generate random parameters
            if 'item' in template['template']:
                params = {
                    'item': random.choice(['apples', 'books', 'pens']),
                    'price': random.randint(1, 20),
                    'quantity': random.randint(2, 10)
                }
                params['total'] = params['price'] * params['quantity']
                params['wrong_total'] = params['price'] + params['quantity']
            elif 'category1' in template['template']:
                params = {
                    'category1': random.choice(['dogs', 'cats', 'birds']),
                    'property': random.choice(['animals', 'pets', 'living']),
                    'item': random.choice(['Rex', 'Fluffy', 'Tweety'])
                }
            elif 'time1' in template['template']:
                hour = random.randint(9, 15)
                duration = random.randint(1, 5)
                params = {
                    'time1': f"{hour}:00",
                    'duration': duration,
                    'correct_time': f"{hour + duration}:00",
                    'incorrect_time': f"{hour - duration}:00"
                }
            
            # Create text
            question = template['template'].format(**params)
            
            if i < size // 2:
                answer = template['correct'].format(**params)
                label = 'correct_reasoning'
            else:
                answer = template['incorrect'].format(**params)
                label = 'incorrect_reasoning'
            
            text = f"{question} {answer}"
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def create_prompt_effectiveness_dataset(self, size: int = 50) -> Tuple[List[str], List[str]]:
        """Create dataset comparing effective vs ineffective prompting"""
        
        base_questions = [
            "Explain the concept of gravity",
            "Describe the process of photosynthesis",
            "What is machine learning?",
            "How does the internet work?",
            "Explain quantum mechanics"
        ]
        
        texts = []
        labels = []
        
        for i in range(size):
            question = base_questions[i % len(base_questions)]
            
            if i < size // 2:
                # Effective prompting (clear, specific, structured)
                prompt = f"Please provide a clear, step-by-step explanation of {question.lower()}. Include the main principles and provide a simple example."
                label = 'effective_prompt'
            else:
                # Ineffective prompting (vague, ambiguous)
                prompt = f"Tell me about {question.lower()} or something."
                label = 'ineffective_prompt'
            
            texts.append(prompt)
            labels.append(label)
        
        return texts, labels
    
    def create_synthetic_graph_dataset(self, size: int = 100) -> Dict[str, Any]:
        """Create synthetic graphs with known spectral properties"""
        
        synthetic_data = []
        
        for i in range(size):
            # Random graph parameters
            seq_len = random.randint(20, 100)
            graph_type = random.choice(['path', 'cycle', 'complete', 'random', 'small_world'])
            noise_level = random.uniform(0, 0.5)
            
            # Generate adjacency matrix based on type
            if graph_type == 'path':
                adj = np.zeros((seq_len, seq_len))
                for j in range(seq_len - 1):
                    adj[j, j + 1] = adj[j + 1, j] = 1
            
            elif graph_type == 'cycle':
                adj = np.zeros((seq_len, seq_len))
                for j in range(seq_len - 1):
                    adj[j, j + 1] = adj[j + 1, j] = 1
                adj[0, seq_len - 1] = adj[seq_len - 1, 0] = 1
            
            elif graph_type == 'complete':
                adj = np.ones((seq_len, seq_len)) - np.eye(seq_len)
            
            elif graph_type == 'random':
                adj = np.random.rand(seq_len, seq_len)
                adj = 0.5 * (adj + adj.T)  # Symmetrize
                adj = (adj > 0.7).astype(float)  # Sparsify
                np.fill_diagonal(adj, 0)
            
            elif graph_type == 'small_world':
                # Watts-Strogatz small world
                k = min(4, seq_len // 4)  # Average degree
                adj = np.zeros((seq_len, seq_len))
                for j in range(seq_len):
                    for offset in range(1, k // 2 + 1):
                        adj[j, (j + offset) % seq_len] = 1
                        adj[j, (j - offset) % seq_len] = 1
                
                # Rewiring with probability
                rewire_prob = 0.3
                for j in range(seq_len):
                    for offset in range(1, k // 2 + 1):
                        if random.random() < rewire_prob:
                            old_neighbor = (j + offset) % seq_len
                            new_neighbor = random.randint(0, seq_len - 1)
                            while new_neighbor == j or adj[j, new_neighbor] == 1:
                                new_neighbor = random.randint(0, seq_len - 1)
                            adj[j, old_neighbor] = 0
                            adj[j, new_neighbor] = 1
                adj = 0.5 * (adj + adj.T)  # Ensure symmetry
            
            # Add noise
            noise = np.random.uniform(0, noise_level, (seq_len, seq_len))
            noise = 0.5 * (noise + noise.T)
            adj += noise
            adj = np.maximum(adj, 0)  # Ensure non-negative
            
            # Compute Laplacian and eigenvalues
            degrees = adj.sum(axis=1)
            laplacian = np.diag(degrees) - adj
            
            try:
                eigenvals, eigenvecs = np.linalg.eigh(laplacian)
                eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            except:
                eigenvals = np.zeros(seq_len)
                eigenvecs = np.eye(seq_len)
            
            # Generate signals with different frequency content
            embedding_dim = 64
            frequency_content = random.choice(['low', 'high', 'mixed', 'uniform'])
            
            if frequency_content == 'low':
                # Low-frequency signal
                k = max(1, seq_len // 4)
                signal = eigenvecs[:, :k] @ np.random.randn(k, embedding_dim)
                expected_smoothness = 'high'
            elif frequency_content == 'high':
                # High-frequency signal
                k = max(1, seq_len // 4)
                signal = eigenvecs[:, -k:] @ np.random.randn(k, embedding_dim)
                expected_smoothness = 'low'
            elif frequency_content == 'mixed':
                # Mixed frequency signal
                k_low = max(1, seq_len // 6)
                k_high = max(1, seq_len // 6)
                signal_low = eigenvecs[:, :k_low] @ np.random.randn(k_low, embedding_dim)
                signal_high = eigenvecs[:, -k_high:] @ np.random.randn(k_high, embedding_dim)
                signal = 0.7 * signal_low + 0.3 * signal_high
                expected_smoothness = 'medium'
            else:  # uniform
                # Uniform across frequencies
                signal = eigenvecs @ np.random.randn(seq_len, embedding_dim)
                expected_smoothness = 'medium'
            
            synthetic_data.append({
                'adjacency': adj,
                'laplacian': laplacian,
                'eigenvalues': eigenvals,
                'eigenvectors': eigenvecs,
                'signal': signal,
                'graph_type': graph_type,
                'seq_len': seq_len,
                'noise_level': noise_level,
                'frequency_content': frequency_content,
                'expected_smoothness': expected_smoothness
            })
        
        return synthetic_data


class BenchmarkEvaluator:
    """Evaluates GSP diagnostics on benchmark tasks"""
    
    def __init__(self, gsp_framework, config: EvaluationConfig):
        self.gsp_framework = gsp_framework
        self.config = config
        self.results = {}
        
    def evaluate_hallucination_detection(self) -> Dict[str, Any]:
        """Evaluate ability to detect hallucinations using GSP diagnostics"""
        logger.info("Evaluating hallucination detection performance...")
        
        # Generate dataset
        dataset_gen = DatasetGenerator(self.config)
        texts, labels = dataset_gen.create_factual_qa_dataset(size=200)
        
        # Extract GSP features
        features, true_labels = self._extract_gsp_features(texts, labels)
        
        # Train classifiers
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.config.random_seed),
            'logistic_regression': LogisticRegression(random_state=self.config.random_seed, max_iter=1000)
        }
        
        results = {}
        for clf_name, clf in classifiers.items():
            clf_results = self._evaluate_classifier(clf, features, true_labels, clf_name)
            results[clf_name] = clf_results
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(features, true_labels)
        results['feature_importance'] = feature_importance
        
        return results
    
    def evaluate_reasoning_quality(self) -> Dict[str, Any]:
        """Evaluate ability to assess reasoning quality"""
        logger.info("Evaluating reasoning quality assessment...")
        
        # Generate reasoning dataset
        dataset_gen = DatasetGenerator(self.config)
        texts, labels = dataset_gen.create_reasoning_dataset(size=200)
        
        # Extract GSP features
        features, true_labels = self._extract_gsp_features(texts, labels)
        
        # Evaluate classification performance
        clf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_seed)
        results = self._evaluate_classifier(clf, features, true_labels, 'reasoning_classifier')
        
        return results
    
    def evaluate_prompt_effectiveness(self) -> Dict[str, Any]:
        """Evaluate ability to distinguish effective vs ineffective prompts"""
        logger.info("Evaluating prompt effectiveness detection...")
        
        # Generate prompt dataset
        dataset_gen = DatasetGenerator(self.config)
        texts, labels = dataset_gen.create_prompt_effectiveness_dataset(size=100)
        
        # Extract GSP features
        features, true_labels = self._extract_gsp_features(texts, labels)
        
        # Evaluate classification performance
        clf = LogisticRegression(random_state=self.config.random_seed, max_iter=1000)
        results = self._evaluate_classifier(clf, features, true_labels, 'prompt_classifier')
        
        return results
    
    def evaluate_synthetic_validation(self) -> Dict[str, Any]:
        """Evaluate on synthetic graphs with known properties"""
        logger.info("Evaluating synthetic graph validation...")
        
        # Generate synthetic dataset
        dataset_gen = DatasetGenerator(self.config)
        synthetic_data = dataset_gen.create_synthetic_graph_dataset(size=100)
        
        results = {
            'frequency_recovery': [],
            'smoothness_prediction': [],
            'graph_type_classification': []
        }
        
        for data in tqdm(synthetic_data, desc="Processing synthetic graphs"):
            # Analyze with GSP framework
            signal_torch = torch.tensor(data['signal'], dtype=torch.float32)
            laplacian_torch = torch.tensor(data['laplacian'], dtype=torch.float32)
            
            diagnostics = self.gsp_framework.spectral_analyzer.analyze_layer(
                signal_torch, laplacian_torch, 0
            )
            
            # Test frequency content recovery
            predicted_smoothness = self._predict_smoothness_category(diagnostics)
            actual_smoothness = data['expected_smoothness']
            
            results['frequency_recovery'].append({
                'predicted': predicted_smoothness,
                'actual': actual_smoothness,
                'correct': predicted_smoothness == actual_smoothness,
                'hfer': diagnostics.hfer,
                'smoothness_index': diagnostics.smoothness_index,
                'spectral_entropy': diagnostics.spectral_entropy
            })
            
            # Test smoothness prediction
            smoothness_score = self._compute_smoothness_score(diagnostics)
            expected_smooth = (actual_smoothness == 'high')
            predicted_smooth = (smoothness_score > 0.5)
            
            results['smoothness_prediction'].append({
                'smoothness_score': smoothness_score,
                'expected_smooth': expected_smooth,
                'predicted_smooth': predicted_smooth,
                'correct': expected_smooth == predicted_smooth
            })
            
            # Graph type classification features
            results['graph_type_classification'].append({
                'graph_type': data['graph_type'],
                'energy': diagnostics.energy,
                'spectral_entropy': diagnostics.spectral_entropy,
                'fiedler_value': diagnostics.fiedler_value,
                'connectivity': diagnostics.connectivity
            })
        
        # Compute summary statistics
        freq_accuracy = np.mean([r['correct'] for r in results['frequency_recovery']])
        smooth_accuracy = np.mean([r['correct'] for r in results['smoothness_prediction']])
        
        results['summary'] = {
            'frequency_recovery_accuracy': freq_accuracy,
            'smoothness_prediction_accuracy': smooth_accuracy,
            'num_samples': len(synthetic_data)
        }
        
        return results
    
    def _extract_gsp_features(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract GSP diagnostic features from texts"""
        features_list = []
        valid_labels = []
        
        for text, label in tqdm(zip(texts, labels), desc="Extracting GSP features", total=len(texts)):
            try:
                # Analyze text
                result = self.gsp_framework.analyze_text(text, save_results=False)
                diagnostics = result['layer_diagnostics']
                
                # Extract features (aggregate across layers)
                layer_features = []
                for diag in diagnostics:
                    layer_features.extend([
                        diag.energy,
                        diag.smoothness_index,
                        diag.spectral_entropy,
                        diag.hfer,
                        diag.fiedler_value,
                        float(diag.connectivity)
                    ])
                
                # Add summary statistics
                energies = [d.energy for d in diagnostics]
                smoothness_indices = [d.smoothness_index for d in diagnostics]
                spectral_entropies = [d.spectral_entropy for d in diagnostics]
                hfers = [d.hfer for d in diagnostics]
                
                summary_features = [
                    np.mean(energies), np.std(energies), np.max(energies),
                    np.mean(smoothness_indices), np.std(smoothness_indices), np.max(smoothness_indices),
                    np.mean(spectral_entropies), np.std(spectral_entropies), np.max(spectral_entropies),
                    np.mean(hfers), np.std(hfers), np.max(hfers),
                    len(diagnostics)  # Number of layers
                ]
                
                features_list.append(layer_features + summary_features)
                valid_labels.append(label)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for text: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid features extracted")
        
        features = np.array(features_list)
        labels_array = np.array(valid_labels)
        
        return features, labels_array
    
    def _evaluate_classifier(self, clf, features: np.ndarray, labels: np.ndarray, 
                           clf_name: str) -> Dict[str, Any]:
        """Evaluate classifier performance with cross-validation"""
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Convert labels to binary if needed
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            labels_binary = np.array([label_map[label] for label in labels])
        else:
            labels_binary = labels
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, 
                           shuffle=True, random_state=self.config.random_seed)
        
        # Compute CV scores
        cv_scores = {}
        for metric in self.config.classification_metrics:
            if metric == 'accuracy':
                scores = cross_val_score(clf, features_scaled, labels_binary, 
                                       cv=cv, scoring='accuracy')
            elif metric == 'precision':
                scores = cross_val_score(clf, features_scaled, labels_binary, 
                                       cv=cv, scoring='precision_macro')
            elif metric == 'recall':
                scores = cross_val_score(clf, features_scaled, labels_binary, 
                                       cv=cv, scoring='recall_macro')
            elif metric == 'f1':
                scores = cross_val_score(clf, features_scaled, labels_binary, 
                                       cv=cv, scoring='f1_macro')
            elif metric == 'auc':
                if len(unique_labels) == 2:
                    scores = cross_val_score(clf, features_scaled, labels_binary, 
                                           cv=cv, scoring='roc_auc')
                else:
                    scores = cross_val_score(clf, features_scaled, labels_binary, 
                                           cv=cv, scoring='roc_auc_ovr_weighted')
            else:
                continue
            
            cv_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
        
        # Fit on full dataset for feature importance
        clf.fit(features_scaled, labels_binary)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(clf, 'feature_importances_'):
            feature_importance = clf.feature_importances_.tolist()
        elif hasattr(clf, 'coef_'):
            feature_importance = np.abs(clf.coef_[0]).tolist()
        
        return {
            'classifier': clf_name,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'num_features': features_scaled.shape[1],
            'num_samples': len(labels_binary),
            'class_distribution': dict(zip(*np.unique(labels_binary, return_counts=True)))
        }
    
    def _analyze_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze which GSP features are most important for classification"""
        
        # Feature names (simplified)
        base_features = ['energy', 'smoothness', 'entropy', 'hfer', 'fiedler', 'connectivity']
        num_layers = (features.shape[1] - 13) // 6  # Subtract summary features
        
        feature_names = []
        for layer in range(num_layers):
            for feat in base_features:
                feature_names.append(f"{feat}_layer_{layer}")
        
        # Add summary feature names
        summary_names = [
            'energy_mean', 'energy_std', 'energy_max',
            'smoothness_mean', 'smoothness_std', 'smoothness_max',
            'entropy_mean', 'entropy_std', 'entropy_max',
            'hfer_mean', 'hfer_std', 'hfer_max',
            'num_layers'
        ]
        feature_names.extend(summary_names)
        
        # Train Random Forest for feature importance
        clf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_seed)
        
        # Convert labels to binary
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            labels_binary = np.array([label_map[label] for label in labels])
        else:
            labels_binary = labels
        
        clf.fit(features, labels_binary)
        
        # Get feature importances
        importances = clf.feature_importances_
        
        # Sort by importance
        importance_indices = np.argsort(importances)[::-1]
        
        top_features = []
        for i in range(min(20, len(importance_indices))):
            idx = importance_indices[i]
            if idx < len(feature_names):
                top_features.append({
                    'feature': feature_names[idx],
                    'importance': importances[idx],
                    'rank': i + 1
                })
        
        return {
            'top_features': top_features,
            'all_importances': importances.tolist(),
            'feature_names': feature_names
        }
    
    def _predict_smoothness_category(self, diagnostics) -> str:
        """Predict smoothness category from diagnostics"""
        hfer = diagnostics.hfer
        smoothness = diagnostics.smoothness_index
        
        if hfer < 0.2 and smoothness < 0.5:
            return 'high'
        elif hfer > 0.6 or smoothness > 1.0:
            return 'low'
        else:
            return 'medium'
    
    def _compute_smoothness_score(self, diagnostics) -> float:
        """Compute normalized smoothness score (0 = rough, 1 = smooth)"""
        # Combine multiple indicators
        hfer_score = 1.0 - min(diagnostics.hfer, 1.0)  # Lower HFER = smoother
        energy_score = 1.0 / (1.0 + diagnostics.smoothness_index)  # Lower SMI = smoother
        entropy_score = 1.0 - min(diagnostics.spectral_entropy / 5.0, 1.0)  # Lower entropy = smoother
        
        # Weighted combination
        smoothness_score = 0.4 * hfer_score + 0.4 * energy_score + 0.2 * entropy_score
        return smoothness_score
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation protocols"""
        logger.info("Starting comprehensive evaluation...")
        
        start_time = time.time()
        all_results = {}
        
        # Hallucination detection
        if 'hallucination_detection' in self.config.benchmark_datasets:
            all_results['hallucination_detection'] = self.evaluate_hallucination_detection()
        
        # Reasoning quality
        if 'reasoning_tasks' in self.config.benchmark_datasets:
            all_results['reasoning_quality'] = self.evaluate_reasoning_quality()
        
        # Prompt effectiveness
        all_results['prompt_effectiveness'] = self.evaluate_prompt_effectiveness()
        
        # Synthetic validation
        if self.config.synthetic_experiments:
            all_results['synthetic_validation'] = self.evaluate_synthetic_validation()
        
        # Computational performance analysis
        if self.config.measure_runtime:
            all_results['computational_analysis'] = self._analyze_computational_performance()
        
        total_time = time.time() - start_time
        all_results['evaluation_metadata'] = {
            'total_time_seconds': total_time,
            'config': asdict(self.config),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Comprehensive evaluation completed in {total_time:.2f} seconds")
        return all_results
    
    def _analyze_computational_performance(self) -> Dict[str, Any]:
        """Analyze computational performance and scalability"""
        logger.info("Analyzing computational performance...")
        
        # Test different sequence lengths
        sequence_lengths = [32, 64, 128, 256, 512]
        performance_data = []
        
        test_text = "This is a test sentence for computational performance analysis. " * 10
        
        for seq_len in sequence_lengths:
            # Truncate text to approximate sequence length
            truncated_text = test_text[:seq_len * 4]  # Rough approximation
            
            # Measure analysis time
            start_time = time.time()
            try:
                result = self.gsp_framework.analyze_text(truncated_text, save_results=False)
                analysis_time = time.time() - start_time
                
                # Count actual tokens
                actual_tokens = len(result['tokens'])
                num_layers = len(result['layer_diagnostics'])
                
                performance_data.append({
                    'target_seq_len': seq_len,
                    'actual_tokens': actual_tokens,
                    'num_layers': num_layers,
                    'analysis_time': analysis_time,
                    'time_per_token': analysis_time / actual_tokens if actual_tokens > 0 else 0,
                    'time_per_layer': analysis_time / num_layers if num_layers > 0 else 0
                })
                
            except Exception as e:
                logger.warning(f"Performance test failed for seq_len {seq_len}: {e}")
                continue
        
        # Compute scaling trends
        if len(performance_data) > 2:
            tokens = [d['actual_tokens'] for d in performance_data]
            times = [d['analysis_time'] for d in performance_data]
            
            # Fit linear scaling model
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(tokens, times)
            
            scaling_analysis = {
                'linear_scaling_slope': slope,
                'linear_scaling_r2': r_value**2,
                'scaling_p_value': p_value
            }
        else:
            scaling_analysis = {}
        
        return {
            'performance_data': performance_data,
            'scaling_analysis': scaling_analysis,
            'device': str(next(self.gsp_framework.instrumenter.model.parameters()).device),
            'model_name': self.gsp_framework.config.model_name
        }


def create_evaluation_report(results: Dict[str, Any], output_dir: str) -> str:
    """Create comprehensive evaluation report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GSP LLM DIAGNOSTICS - COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Metadata
    if 'evaluation_metadata' in results:
        metadata = results['evaluation_metadata']
        report_lines.append("EVALUATION METADATA")
        report_lines.append("-" * 40)
        report_lines.append(f"Total evaluation time: {metadata['total_time_seconds']:.2f} seconds")
        report_lines.append(f"Timestamp: {metadata['timestamp']}")
        report_lines.append(f"Model: {metadata['config']['model_name'] if 'model_name' in metadata['config'] else 'Unknown'}")
        report_lines.append("")
    
    # Hallucination detection results
    if 'hallucination_detection' in results:
        report_lines.append("HALLUCINATION DETECTION PERFORMANCE")
        report_lines.append("-" * 50)
        
        hd_results = results['hallucination_detection']
        for clf_name, clf_results in hd_results.items():
            if clf_name == 'feature_importance':
                continue
            
            report_lines.append(f"\n{clf_name.upper()}:")
            cv_scores = clf_results.get('cv_scores', {})
            for metric, scores in cv_scores.items():
                mean_score = scores['mean']
                std_score = scores['std']
                report_lines.append(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Feature importance
        if 'feature_importance' in hd_results:
            fi = hd_results['feature_importance']
            report_lines.append(f"\nTop 5 Most Important Features:")
            for feat in fi['top_features'][:5]:
                report_lines.append(f"  {feat['rank']}. {feat['feature']}: {feat['importance']:.4f}")
        
        report_lines.append("")
    
    # Synthetic validation results
    if 'synthetic_validation' in results:
        report_lines.append("SYNTHETIC VALIDATION PERFORMANCE")
        report_lines.append("-" * 40)
        
        sv_results = results['synthetic_validation']
        summary = sv_results.get('summary', {})
        
        freq_acc = summary.get('frequency_recovery_accuracy', 0)
        smooth_acc = summary.get('smoothness_prediction_accuracy', 0)
        
        report_lines.append(f"Frequency recovery accuracy: {freq_acc:.4f}")
        report_lines.append(f"Smoothness prediction accuracy: {smooth_acc:.4f}")
        report_lines.append(f"Number of synthetic samples: {summary.get('num_samples', 0)}")
        report_lines.append("")
    
    # Computational performance
    if 'computational_analysis' in results:
        report_lines.append("COMPUTATIONAL PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        
        comp_results = results['computational_analysis']
        scaling = comp_results.get('scaling_analysis', {})
        
        if 'linear_scaling_slope' in scaling:
            slope = scaling['linear_scaling_slope']
            r2 = scaling['linear_scaling_r2']
            report_lines.append(f"Linear scaling slope: {slope:.6f} seconds/token")
            report_lines.append(f"Scaling R²: {r2:.4f}")
        
        perf_data = comp_results.get('performance_data', [])
        if perf_data:
            avg_time_per_token = np.mean([d['time_per_token'] for d in perf_data])
            report_lines.append(f"Average time per token: {avg_time_per_token:.4f} seconds")
        
        report_lines.append("")
    
    # Summary and conclusions
    report_lines.append("SUMMARY AND CONCLUSIONS")
    report_lines.append("-" * 40)
    
    # Extract key metrics for summary
    best_performance = 0
    best_task = "unknown"
    
    if 'hallucination_detection' in results:
        hd_results = results['hallucination_detection']
        for clf_name, clf_results in hd_results.items():
            if isinstance(clf_results, dict) and 'cv_scores' in clf_results:
                accuracy = clf_results['cv_scores'].get('accuracy', {}).get('mean', 0)
                if accuracy > best_performance:
                    best_performance = accuracy
                    best_task = f"hallucination detection ({clf_name})"
    
    report_lines.append(f"Best performance: {best_performance:.4f} on {best_task}")
    
    if best_performance > 0.8:
        report_lines.append("• Excellent performance - GSP diagnostics show strong predictive power")
    elif best_performance > 0.7:
        report_lines.append("• Good performance - GSP diagnostics are useful for LLM monitoring")
    elif best_performance > 0.6:
        report_lines.append("• Moderate performance - GSP diagnostics show some predictive value")
    else:
        report_lines.append("• Limited performance - further research needed")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = Path(output_dir) / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Evaluation report saved to {report_path}")
    return report_text


def main():
    """Main function for running evaluations"""
    import argparse
    from gsp_llm_diagnostics import GSPDiagnosticsFramework, GSPConfig
    
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation of GSP LLM Diagnostics")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", 
                       choices=['factual_qa', 'hallucination_detection', 'reasoning_tasks'],
                       default=['hallucination_detection', 'reasoning_tasks'],
                       help="Datasets to evaluate on")
    parser.add_argument("--no_synthetic", action="store_true", help="Skip synthetic experiments")
    parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
    
    args = parser.parse_args()
    
    # Create configurations
    gsp_config = GSPConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        verbose=True
    )
    
    eval_config = EvaluationConfig(
        benchmark_datasets=args.datasets,
        synthetic_experiments=not args.no_synthetic,
        cross_validation_folds=args.cv_folds
    )
    
    # Run evaluation
    with GSPDiagnosticsFramework(gsp_config) as gsp_framework:
        gsp_framework.instrumenter.load_model(args.model_name)
        
        evaluator = BenchmarkEvaluator(gsp_framework, eval_config)
        results = evaluator.run_comprehensive_evaluation()
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create report
        report = create_evaluation_report(results, args.output_dir)
        
        print("Evaluation Complete!")
        print(f"Results saved to: {args.output_dir}")
        print("\nReport Summary:")
        print(report)


if __name__ == "__main__":
    main()