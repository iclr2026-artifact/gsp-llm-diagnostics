#!/usr/bin/env python3
"""
Command Line Interface for GSP LLM Diagnostics Framework
Provides easy-to-use CLI for running various analysis tasks with modern language models
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch

# Import our modules
from gsp_llm_diagnostics import GSPDiagnosticsFramework, GSPConfig
#from cli_modules.robustness_analysis import RobustnessAnalyzer, PerturbationConfig
#from cli_modules.evaluation_benchmarks import BenchmarkEvaluator, EvaluationConfig, create_evaluation_report

# --- UTF-8 console safety on Windows ---
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern model configurations with optimal settings
MODERN_MODELS = {
    # Llama models
    'llama-3.2-1b': {
        'name': 'meta-llama/Llama-3.2-1B',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.2-3b': {
        'name': 'meta-llama/Llama-3.2-3B',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.2-3b-instruct': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.1-8b': {
        'name': 'meta-llama/Meta-Llama-3.1-8B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.1-70b': {
        'name': 'meta-llama/Meta-Llama-3.1-70B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    
    # Mistral models
    'mistral-7b': {
        'name': 'mistralai/Mistral-7B-v0.1',
        'max_length': 8192,
        'torch_dtype': 'float16'
    },
    'mistral-7b-instruct': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'max_length': 8192,
        'torch_dtype': 'float16'
    },
    'mixtral-8x7b': {
        'name': 'mistralai/Mixtral-8x7B-v0.1',
        'max_length': 8192,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    'mixtral-8x22b': {
        'name': 'mistralai/Mixtral-8x22B-v0.1',
        'max_length': 8192,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    
    # Qwen models
    'qwen2-7b': {
        'name': 'Qwen/Qwen2-7B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'qwen2-72b': {
        'name': 'Qwen/Qwen2-72B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    'qwen2.5-7b': {
        'name': 'Qwen/Qwen2.5-7B',
        'max_length': 8192,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'qwen2.5-72b': {
        'name': 'Qwen/Qwen2.5-72B',
        'max_length': 8192,
        'trust_remote_code': True,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    
    # Gemma models
    'gemma-2b': {
        'name': 'google/gemma-2b',
        'max_length': 2048,
        'torch_dtype': 'float16'
    },
    'gemma-7b': {
        'name': 'google/gemma-7b',
        'max_length': 2048,
        'torch_dtype': 'float16'
    },
    'gemma2-9b': {
        'name': 'google/gemma-2-9b',
        'max_length': 4096,
        'torch_dtype': 'float16'
    },
    'gemma2-27b': {
        'name': 'google/gemma-2-27b',
        'max_length': 4096,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    
    # Phi models
    'phi-3-mini': {
        'name': 'microsoft/Phi-3-mini-4k-instruct',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'phi-3-medium': {
        'name': 'microsoft/Phi-3-medium-4k-instruct',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'phi-3.5-mini': {
        'name': 'microsoft/Phi-3.5-mini-instruct',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    
    # Legacy models for comparison
    'gpt2': {
        'name': 'gpt2',
        'max_length': 1024,
        'torch_dtype': 'float32'
    },
    'gpt2-medium': {
        'name': 'gpt2-medium',
        'max_length': 1024,
        'torch_dtype': 'float32'
    },
    'gpt2-large': {
        'name': 'gpt2-large',
        'max_length': 1024,
        'torch_dtype': 'float32'
    },
    'distilgpt2': {
        'name': 'distilgpt2',
        'max_length': 1024,
        'torch_dtype': 'float32'
    },
    
    # Additional modern models
    'falcon-7b': {
        'name': 'tiiuae/falcon-7b',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'vicuna-7b': {
        'name': 'lmsys/vicuna-7b-v1.5',
        'max_length': 2048,
        'torch_dtype': 'float16'
    },
    'openchat-7b': {
        'name': 'openchat/openchat-3.5-0106',
        'max_length': 8192,
        'torch_dtype': 'float16'
    },

    'chatglm3-6b': {
        'name': 'THUDM/chatglm3-6b',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },

    'chatglm2-6b': {
        'name': 'THUDM/chatglm2-6b',
        'max_length': 8192,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },

    # Baichuan models
    'baichuan2-7b': {
        'name': 'baichuan-inc/Baichuan2-7B-Base',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'baichuan2-13b': {
        'name': 'baichuan-inc/Baichuan2-13B-Base',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },

    # Yi models (01.AI)
    'yi-6b': {
        'name': '01-ai/Yi-6B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'yi-34b': {
        'name': '01-ai/Yi-34B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },

    # InternLM models (Shanghai AI Lab)
    'internlm-7b': {
        'name': 'internlm/internlm-7b',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
}

def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model configuration by key or return custom config for unknown models"""
    if model_key in MODERN_MODELS:
        return MODERN_MODELS[model_key]
    else:
        # Assume it's a HuggingFace model path
        return {
            'name': model_key,
            'max_length': 2048,
            'torch_dtype': 'float16',
            'trust_remote_code': False
        }

def setup_argument_parser():
    """Setup comprehensive argument parser with modern model support"""
    parser = argparse.ArgumentParser(
        description="Graph Signal Processing Framework for LLM Diagnostics - Modern Models Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Analyze with modern models
  python cli.py analyze --text "The capital of France is Paris." --model llama-3.2-3b
  python cli.py analyze --text "Explain quantum computing." --model mistral-7b
  python cli.py compare --models llama-3.2-1b qwen2-7b phi-3-mini --text "Hello world"
  
  # Legacy model support
  python cli.py analyze --text "Test text" --model gpt2
  
  # Custom HuggingFace model
  python cli.py analyze --text "Test" --model microsoft/DialoGPT-medium
  
  # Batch analysis with large model
  python cli.py batch --config config.json --input_dir ./data --model llama-3.1-8b
  
Available model shortcuts:
{', '.join(sorted(MODERN_MODELS.keys()))}
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text(s) with GSP diagnostics')
    add_analyze_arguments(analyze_parser)
    
    # Robustness command
    robustness_parser = subparsers.add_parser('robustness', help='Run robustness analysis')
    add_robustness_arguments(robustness_parser)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Run benchmark evaluation')
    add_evaluation_arguments(evaluate_parser)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch processing with custom configuration')
    add_batch_arguments(batch_parser)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare different models or configurations')
    add_compare_arguments(compare_parser)
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available model shortcuts')
    add_list_models_arguments(list_parser)
    
    return parser

def generate_response(hf_model, hf_tokenizer, prompt_text, max_new_tokens=20):
    """Generate response with minimal risk of CUDA assertion errors"""
    try:
        # Use very conservative settings
        inputs = hf_tokenizer(
            prompt_text, 
            return_tensors="pt",
            max_length=256,  # Keep it short
            truncation=True,
            padding=False,  # No padding for generation
            add_special_tokens=True
        )
        
        # Move to device
        model_device = next(hf_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Very simple generation
        with torch.no_grad():
            outputs = hf_model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=hf_tokenizer.eos_token_id,
                eos_token_id=hf_tokenizer.eos_token_id,
                use_cache=False  # Disable KV cache to avoid issues
            )
        
        # Decode new tokens only
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return ""


def add_analyze_arguments(parser):
    """Add arguments for analyze command with modern model defaults"""
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Single text to analyze')
    input_group.add_argument('--text_file', type=str, help='Path to a UTF-8 text file (alternative to --text)')
    input_group.add_argument('--input_file', type=str, help='File containing texts (one per line)')
    
    # Model options with modern defaults
    parser.add_argument('--model', type=str, default='llama-3.2-1b', 
                       help='Model name or shortcut (default: llama-3.2-1b). Use "list-models" to see available shortcuts')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum sequence length (default: model-specific)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'], help='Device to use')
    parser.add_argument('--precision', type=str, default='auto',
                       choices=['auto', 'float32', 'float16', 'bfloat16'], 
                       help='Model precision (default: auto)')
    parser.add_argument('--use_flash_attention', action='store_true',
                       help='Use Flash Attention 2 for supported models')
    parser.add_argument('--load_in_8bit', action='store_true',
                       help='Load model in 8-bit precision (requires bitsandbytes)')
    parser.add_argument('--load_in_4bit', action='store_true',
                       help='Load model in 4-bit precision (requires bitsandbytes)')
    parser.add_argument('--emit_text', action='store_true',
                    help='Also run model.generate() and save a response (SLOW). Off by default.')
    parser.add_argument('--analyze_with_response', action='store_true',
                        help='If --emit_text is set, append the generated response to the analyzed text. Off by default.')

    
    # GSP options
    parser.add_argument('--head_aggregation', type=str, default='uniform',
                       choices=['uniform', 'attention_weighted', 'learnable'],
                       help='Attention head aggregation method')
    parser.add_argument('--symmetrization', type=str, default='symmetric',
                       choices=['symmetric', 'row_norm', 'col_norm'],
                       help='Attention symmetrization method')
    parser.add_argument('--normalization', type=str, default='rw',
                       choices=['rw', 'sym', 'none'], help='Laplacian normalization')
    parser.add_argument('--hfer_cutoff', type=float, default=0.1,
                       help='High-frequency energy ratio cutoff (default: 0.1)')
    
    # Add these arguments after your existing ones in add_analyze_arguments():
    parser.add_argument('--gold', type=str, default=None,
                    help='Gold/expected answer (yes/no, single letter, or number)')
    parser.add_argument('--style', type=str, default=None,
                    help='Prompt style tag (standard/cot/tot/cod) to log with outputs')
    parser.add_argument('--qid', type=str, default=None,
                        help='Example id to record with outputs')
    parser.add_argument('--perlayer_out', type=str, default=None,
                        help='CSV path to append per-layer metrics for aggregation')
    
    # Analysis options optimized for modern models
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of layers to analyze (default: all)')
    parser.add_argument('--num_eigenvalues', type=int, default=100,
                       help='Number of eigenvalues to compute (default: 100)')
    parser.add_argument('--eigen_solver', type=str, default='sparse',
                       choices=['sparse', 'dense'], help='Eigenvalue solver')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing (default: 1)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./gsp_results',
                       help='Output directory (default: ./gsp_results)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save diagnostic plots')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # --- Causal intervention flags ---
    parser.add_argument('--ablate_heads', type=str, default=None,
                        help='Head ablation spec, e.g. "2:0-3; 6:5" (layer:headIdx[,..][;...])')
    parser.add_argument('--patch_layer', type=int, default=None,
                        help='Layer index for residual patching')
    parser.add_argument('--patch_t_lo', type=int, default=None,
                        help='Token start (inclusive) for patch span')
    parser.add_argument('--patch_t_hi', type=int, default=None,
                        help='Token end (exclusive) for patch span')
    parser.add_argument('--donor_text', type=str, default=None,
                        help='Donor text for patching (residual slice source)')
    parser.add_argument('--donor_text_file', type=str, default=None,
                        help='UTF-8 file with donor text')


def add_robustness_arguments(parser):
    """Add arguments for robustness command"""
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--model', type=str, default='llama-3.2-1b', 
                       help='Model name or shortcut')
    
    # Perturbation options
    parser.add_argument('--perturbation_types', nargs='+', 
                       choices=['token_swap', 'token_delete', 'token_insert', 'embedding_noise'],
                       default=['token_swap', 'token_delete', 'embedding_noise'],
                       help='Types of perturbations to apply')
    parser.add_argument('--num_perturbations', type=int, default=10,
                       help='Number of perturbations per type')
    parser.add_argument('--noise_levels', nargs='+', type=float, 
                       default=[0.01, 0.05, 0.1, 0.2, 0.5],
                       help='Noise levels for embedding perturbations')
    
    # Model options
    parser.add_argument('--precision', type=str, default='auto',
                       choices=['auto', 'float32', 'float16', 'bfloat16'])
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./robustness_results',
                       help='Output directory')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create robustness visualizations')


def add_evaluation_arguments(parser):
    """Add arguments for evaluation command"""
    parser.add_argument('--model', type=str, default='llama-3.2-1b', 
                       help='Model to evaluate')
    
    # Dataset options
    parser.add_argument('--datasets', nargs='+',
                       choices=['factual_qa', 'hallucination_detection', 'reasoning_tasks', 'multilingual'],
                       default=['hallucination_detection'],
                       help='Benchmark datasets to evaluate on')
    parser.add_argument('--dataset_size', type=int, default=200,
                       help='Size of generated datasets')
    
    # Evaluation options
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Cross-validation folds')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--no_synthetic', action='store_true',
                       help='Skip synthetic experiments')
    
    # Performance analysis
    parser.add_argument('--measure_runtime', action='store_true',
                       help='Measure computational performance')
    parser.add_argument('--scalability_analysis', action='store_true',
                       help='Run scalability analysis')
    parser.add_argument('--memory_profiling', action='store_true',
                       help='Profile memory usage')
    
    # Model options
    parser.add_argument('--precision', type=str, default='auto')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory')


def add_batch_arguments(parser):
    """Add arguments for batch command"""
    parser.add_argument('--config', type=str, required=True,
                       help='JSON configuration file')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing text files')
    parser.add_argument('--output_dir', type=str, default='./batch_results',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='llama-3.2-1b',
                       help='Model name or shortcut to override config')
    parser.add_argument('--file_pattern', type=str, default='*.txt',
                       help='File pattern to match (default: *.txt)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum worker processes for parallel processing')
    parser.add_argument('--precision', type=str, default='auto')


def add_compare_arguments(parser):
    """Add arguments for compare command"""
    parser.add_argument('--models', nargs='+', required=True,
                       help='Models to compare (use shortcuts or full names)')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to analyze with all models')
    parser.add_argument('--metrics', nargs='+',
                       choices=['energy', 'smoothness', 'entropy', 'hfer', 'fiedler', 'all'],
                       default=['all'], help='Metrics to compare')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Output directory')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create comparison plots')
    parser.add_argument('--precision', type=str, default='auto')
    parser.add_argument('--load_in_8bit', action='store_true')


def add_list_models_arguments(parser):
    """Add arguments for list-models command"""
    parser.add_argument('--category', type=str, choices=['all', 'llama', 'mistral', 'qwen', 'gemma', 'phi', 'legacy'],
                       default='all', help='Filter models by category')
    parser.add_argument('--show_details', action='store_true',
                       help='Show detailed model information')


def cmd_list_models(args):
    """Handle list-models command"""
    if args.category == 'all':
        models_to_show = MODERN_MODELS
    else:
        models_to_show = {k: v for k, v in MODERN_MODELS.items() 
                         if args.category in k}
    
    print(f"\nAvailable Model Shortcuts ({args.category}):")
    print("=" * 60)
    
    for shortcut, config in sorted(models_to_show.items()):
        if args.show_details:
            print(f"{shortcut:20} -> {config['name']}")
            print(f"{'':20}    Max Length: {config['max_length']}")
            print(f"{'':20}    Precision: {config.get('torch_dtype', 'float32')}")
            if 'device_map' in config:
                print(f"{'':20}    Multi-GPU: Yes")
            print()
        else:
            print(f"{shortcut:20} -> {config['name']}")
    
    print(f"\nTotal: {len(models_to_show)} models")
    print("\nUsage: python cli.py analyze --model <shortcut> --text \"Your text here\"")
    if not args.show_details:
        print("Use --show_details for more information about each model")

import re

import re

import re

def extract_answer(text: str) -> str:
    """
    Extract answers from model responses for simple comparison/math questions.
    Focuses on single letters, numbers, and simple words.
    """
    if not text:
        return "unknown"

    text = text.strip()
    
    # 1) Look for explicit answer patterns first
    answer_patterns = [
        r'(?:answer|result|solution)\s+is\s+([A-Z]|\d+|[a-zA-Z]+)',
        r'(?:therefore|so|thus),?\s+([A-Z])\s+is',
        r'(?:the\s+)?(?:answer|result)\s+is\s+([A-Z]|\d+)',
        r'([A-Z])\s+is\s+the\s+largest',
        r'([A-Z])\s+is\s+larger',
        r'([A-Z])\s+is\s+greater',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if len(answer) == 1 and answer.isalpha():
                return answer.upper()
            elif answer.isdigit():
                return answer
            elif answer.lower() in ['yes', 'no']:
                return answer.lower()
    
    # 2) For math questions, look for numbers
    if any(word in text.lower() for word in ['equals', 'is equal', '+', '-', '*', '/']):
        # Find "X equals Y" or "X is Y" patterns
        math_patterns = [
            r'(?:equals?|is)\s+(\d+)',
            r'(\d+)\s*$',  # Number at end
        ]
        for pattern in math_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    
    # 3) For comparison questions (A > B type), look for single letters
    if '>' in text or 'larger' in text.lower() or 'greatest' in text.lower():
        # Look for single capital letters that appear as standalone answers
        letters = re.findall(r'\b([A-Z])\b', text)
        if letters:
            # Often the first letter mentioned is the answer for these types
            return letters[0]
    
    # 4) Look for yes/no answers
    text_lower = text.lower()
    if 'yes' in text_lower and 'no' not in text_lower:
        return 'yes'
    elif 'no' in text_lower and 'yes' not in text_lower:
        return 'no'
    
    # 5) For city/country questions, look for proper nouns
    if any(word in text.lower() for word in ['capital', 'city', 'country']):
        # Look for capitalized words that are likely place names
        places = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
        known_places = ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Madrid', 'Washington']
        for place in places:
            if place in known_places:
                return place
        if places:
            return places[0]
    
    # 6) Last resort: look for any single letter or number
    # Check first few sentences for single letters/numbers
    first_part = '. '.join(text.split('. ')[:3])  # First 3 sentences
    
    single_items = re.findall(r'\b([A-Z]|\d+)\b', first_part)
    if single_items:
        return single_items[0]
    
    return "unknown"

def create_model_config(model_key: str, args) -> GSPConfig:
    """Create GSPConfig with model-specific settings - compatible with existing framework"""
    model_info = get_model_info(model_key)
    
    # Determine device
    if hasattr(args, 'device') and args.device != 'auto':
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Determine max length
    max_length = getattr(args, 'max_length', None)
    if max_length is None:
        max_length = model_info['max_length']
    
    # Only use parameters that your existing GSPConfig supports
    return GSPConfig(
        model_name=model_info['name'],  # Use the full HuggingFace name
        max_length=max_length,
        device=device,
        # Remove all the unsupported parameters:
        # torch_dtype, trust_remote_code, device_map, load_in_8bit, etc.
        head_aggregation=getattr(args, 'head_aggregation', 'uniform'),
        symmetrization=getattr(args, 'symmetrization', 'symmetric'),
        normalization=getattr(args, 'normalization', 'rw'),
        hfer_cutoff_ratio=getattr(args, 'hfer_cutoff', 0.1),
        num_layers_analyze=getattr(args, 'num_layers', None),
        num_eigenvalues=getattr(args, 'num_eigenvalues', 100),
        eigen_solver=getattr(args, 'eigen_solver', 'sparse'),
        output_dir=getattr(args, 'output_dir', './gsp_results'),
        save_plots=getattr(args, 'save_plots', False),
        save_intermediate=getattr(args, 'save_intermediate', False),
        verbose=getattr(args, 'verbose', False)
    )

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HeadSpec:
    layer: int
    heads: List[int]

def parse_heads(spec: Optional[str]) -> List[HeadSpec]:
    if not spec or not spec.strip():
        return []
    out: List[HeadSpec] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        L_str, hs = chunk.split(":")
        L = int(L_str.strip())
        idxs = []
        for tok in hs.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-"); a, b = int(a), int(b)
                idxs.extend(range(a, b + 1))
            else:
                idxs.append(int(tok))
        out.append(HeadSpec(L, sorted(set(idxs))))
    return out

def build_head_mask(config, specs: List[HeadSpec], device: str):
    nL = getattr(config, "num_hidden_layers", None)
    nH = getattr(config, "num_attention_heads", None)
    if nL is None or nH is None or not specs:
        return None
    mask = torch.ones(nL, nH, dtype=torch.float32, device=device)
    for s in specs:
        if 0 <= s.layer < nL:
            for h in s.heads:
                if 0 <= h < nH:
                    mask[s.layer, h] = 0.0
    return mask

class PatchResidualAt:
    """
    Forward hook to patch residual slice at a layer and token window.
    Assumes transformer blocks are at model.model.layers[idx].
    """
    def __init__(self, t_lo: int, t_hi: int, donor_hidden: torch.Tensor):
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.donor_hidden = donor_hidden  # (1, seq, dim)

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            hs = out[0]; rest = out[1:]
        else:
            hs = out; rest = None
        L = hs.shape[1]
        a = max(0, min(self.t_lo, L))
        b = max(0, min(self.t_hi, L))
        if a < b:
            hs = hs.clone()
            hs[:, a:b, :] = self.donor_hidden[:, a:b, :].to(hs.device)
        return (hs, *rest) if rest is not None else hs

# --- Head ablation via q_proj/v_proj forward hooks (works on LLaMA/Qwen) ---
from typing import Dict, List, Tuple

def _layer_modules_for_llama_like(hf_model, layer_idx: int):
    """
    Returns (q_proj_module, v_proj_module) for decoder-only LLaMA/Qwen-style blocks.
    Adjust here if your framework exposes different names.
    """
    block = hf_model.model.layers[layer_idx]
    attn = getattr(block, "self_attn", getattr(block, "attention", None))
    if attn is None:
        raise RuntimeError(f"Layer {layer_idx} has no self_attn/attention module")
    q_proj = getattr(attn, "q_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if q_proj is None or v_proj is None:
        raise RuntimeError(f"Layer {layer_idx} missing q_proj/v_proj")
    return q_proj, v_proj

# --- Head ablation via q_proj/v_proj forward hooks (robust on LLaMA/Qwen/Mistral) ---
from typing import List, Optional

def _get_attn_module_for_layer(hf_model, layer_idx: int):
    """
    Returns the attention module of a decoder block in common HF decoders.
    Adjust here if your model has a different structure.
    """
    # Most decoder-only models expose .model.layers
    block = getattr(hf_model, "model", hf_model).layers[layer_idx]
    # Common attribute names across families
    for name in ("self_attn", "attention", "attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise RuntimeError(f"Layer {layer_idx} has no self-attn module")

def _get_qv_linears(attn_mod):
    """
    Returns (q_proj, v_proj) Linear modules.
    Handles LLaMA/Qwen/Mistral-like attention blocks.
    """
    # LLaMA/Qwen/Mistral often have q_proj/v_proj explicitly:
    q_proj = getattr(attn_mod, "q_proj", None)
    v_proj = getattr(attn_mod, "v_proj", None)
    if q_proj is not None and v_proj is not None:
        return q_proj, v_proj

    # Some variants pack QKV together (rare for these families). If you hit this,
    # you'll need to split channels; for now, fail loudly.
    raise RuntimeError("Attention module lacks q_proj/v_proj linears; add a splitter for packed QKV here.")

def _make_proj_hook(heads_to_zero: List[int], head_dim: int):
    """
    Hook that zeros the per-head slices in the *output* of q_proj/v_proj.
    output shape: (batch, seq, n_heads*head_dim)
    """
    head_slices = [(h * head_dim, (h + 1) * head_dim) for h in sorted(set(heads_to_zero))]
    def hook(module, inputs, output):
        if output is None:
            return output
        y = output
        for a, b in head_slices:
            y[..., a:b] = 0
        return y
    return hook

def attach_qv_ablation_hooks(hf_model, specs: List["HeadSpec"]) -> List:
    """
    Attach forward hooks on q_proj and v_proj for each (layer, heads).
    Returns a list of handles; caller must .remove() them.
    """
    handles = []
    nH = hf_model.config.num_attention_heads
    head_dim = hf_model.config.hidden_size // nH

    for s in specs:
        attn = _get_attn_module_for_layer(hf_model, s.layer)
        q_proj, v_proj = _get_qv_linears(attn)
        heads = [h for h in s.heads if 0 <= h < nH]
        if not heads:
            continue
        hook = _make_proj_hook(heads, head_dim)
        handles.append(q_proj.register_forward_hook(hook))
        handles.append(v_proj.register_forward_hook(hook))
    return handles

def _get_attn_module_for_layer_generic(hf_model, layer_idx: int):
    """
    Return the attention module for a decoder block across families (LLaMA/Qwen/Phi/Mistral).
    """
    block_container = getattr(hf_model, "model", hf_model)
    block = block_container.layers[layer_idx]
    for name in ("self_attn", "attention", "attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise RuntimeError(f"Layer {layer_idx} has no attention module")

def _attach_attn_output_head_zero_hook(hf_model, specs: List[HeadSpec]) -> List:
    """
    Forward-hook on the ATTENTION MODULE OUTPUT to zero selected heads.
    Works even when q_proj/v_proj are packed/absent (e.g., Phi-3).
    """
    handles = []
    nH = hf_model.config.num_attention_heads
    Hd = hf_model.config.hidden_size // nH

    # Which heads to zero per layer
    by_layer = {}
    for s in specs:
        valid = [h for h in s.heads if 0 <= h < nH]
        if valid:
            by_layer.setdefault(s.layer, sorted(set(valid)))

    if not by_layer:
        return handles

    for layer_idx, heads in by_layer.items():
        attn = _get_attn_module_for_layer_generic(hf_model, layer_idx)
        head_slices = [(h * Hd, (h + 1) * Hd) for h in heads]

        def hook(module, inputs, output, head_slices=head_slices, Hd=Hd):
            """
            `output` here is the ATTENTION MODULE OUTPUT (already projected back to hidden size).
            We zero contiguous slices corresponding to heads.
            Shape is expected (B, T, hidden), hidden = nH * Hd.
            """
            y = output
            # some impls may return tuple; ensure we operate on the tensor
            if isinstance(y, tuple):
                y = y[0]
            if y is None:
                return output
            # y: (B, T, hidden)
            for a, b in head_slices:
                y[..., a:b] = 0
            return y

        handles.append(attn.register_forward_hook(hook))
    return handles


def setup_model_for_analysis(hf_model):
    """
    Setup model configuration for better compatibility with GSP analysis.
    Handles common issues with Qwen2 and similar models.
    """
    if hasattr(hf_model, "config"):
        # Force attention outputs
        hf_model.config.output_attentions = True
        hf_model.config.output_hidden_states = True
        
        # Use eager attention to avoid SDPA issues
        try:
            hf_model.config.attn_implementation = "eager"
        except Exception:
            pass
    
    return hf_model


def safe_tokenize_with_fallback(tokenizer, text: str, **kwargs):
    """
    Safely tokenize text with fallbacks for edge cases that produce zero tokens.
    Common with some Qwen2 configurations.
    """
    # Try original text first
    enc = tokenizer(text, **kwargs)
    
    # Check if we got valid tokens
    if (enc.get("input_ids", None) is None or 
        enc["input_ids"].numel() == 0 or 
        enc["input_ids"].shape[1] == 0):
        
        # Try various fallbacks
        candidates = [
            text.strip(),
            " " + (text.strip() or ""),
            (text.strip() + " .") if text.strip() else ".",
            "Hello world."  # Ultimate fallback
        ]
        
        for candidate in candidates:
            try:
                enc = tokenizer(candidate, **kwargs)
                if (enc.get("input_ids", None) is not None and 
                    enc["input_ids"].numel() > 0 and 
                    enc["input_ids"].shape[1] > 0):
                    return enc, candidate
            except Exception:
                continue
        
        raise RuntimeError("Failed to tokenize any variant of the input text")
    
    return enc, text


def cmd_analyze(args):
    """Analyze (single or batch), generate a model answer, score vs gold, and log per-layer metrics."""
    logger.info(f"Starting GSP analysis with model: {args.model}")

    # Build config & open framework
    config = create_model_config(args.model, args)

    # Helper: normalize gold labels
    def _normalize_gold(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s = s.strip()
        lo = s.lower()
        if lo in ("yes", "no"):
            return lo
        if len(s) == 1 and s.isalpha():
            return s.upper()
        try:
            if "." in s:
                return str(float(s)).rstrip("0").rstrip(".")
            return str(int(s))
        except Exception:
            return s

    # Single text if provided
    single_text: Optional[str] = None
    if args.text_file:
        try:
            single_text = Path(args.text_file).read_text(encoding='utf-8').strip()
            if not single_text:
                logger.error(f"Text file {args.text_file} is empty")
                return
        except Exception as e:
            logger.error(f"Failed to read text file {args.text_file}: {e}")
            return
    elif args.text:
        single_text = args.text.strip()

    # Donor text for patching (optional)
    donor_text: Optional[str] = None
    if args.donor_text_file:
        try:
            donor_text = Path(args.donor_text_file).read_text(encoding='utf-8').strip()
        except Exception as e:
            logger.error(f"Failed to read donor text file: {e}")
            return
    elif args.donor_text:
        donor_text = args.donor_text.strip()

    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            framework.instrumenter.register_hooks()

            # HF model + tokenizer
            try:
                hf_model = framework.instrumenter.model
                hf_tokenizer = framework.instrumenter.tokenizer
            except Exception as e:
                raise RuntimeError("Could not access HF model/tokenizer from instrumenter") from e

            hf_model = setup_model_for_analysis(hf_model)
            device = next(hf_model.parameters()).device

            # Optional: head ablation hooks
            ablation_handles = []
            if args.ablate_heads:
                try:
                    head_specs = parse_heads(args.ablate_heads)
                    if head_specs:
                        logger.info(f"Setting up head ablation for {len(head_specs)} layer specifications")
                        ablation_handles = _attach_attn_output_head_zero_hook(hf_model, head_specs)
                except Exception as e:
                    logger.warning(f"Failed to setup head ablation: {e}")

            # ========= inner runner for one prompt =========
            def run_one(prompt_text: str, qid: Optional[str], gold_raw: Optional[str]):
                # Tokenize (with fallbacks)
                try:
                    enc, processed_text = safe_tokenize_with_fallback(
                        hf_tokenizer, prompt_text, add_special_tokens=True, return_tensors="pt"
                    )
                    if processed_text != prompt_text:
                        logger.warning(f"Text modified for tokenization: '{prompt_text[:80]}' -> '{processed_text[:80]}'")
                        prompt_text = processed_text
                except Exception as e:
                    logger.error(f"Tokenization failed: {e}")
                    return

                # Generate model response
                generated_response = ""
                prediction = "unknown"
                gold = _normalize_gold(gold_raw)
                is_correct = None

                if args.emit_text:
                    try:
                        logger.info("Generating model response...")
                        generated_response = generate_response(hf_model, hf_tokenizer, prompt_text, max_new_tokens=50)
                        prediction = extract_answer(generated_response)
                        if gold is not None:
                            is_correct = (prediction == gold)

                        logger.info(f"Generated response: {generated_response[:120]}...")
                        logger.info(f"Extracted prediction: {prediction}")

                        pred_record = {
                            "qid": qid,
                            "style": getattr(args, "style", None),
                            "prompt": prompt_text,
                            "response": generated_response,
                            "prediction": prediction,
                            "gold": gold,
                            "model": args.model,
                            "is_correct": (bool(is_correct) if gold is not None else None)
                        }
                        out_dir = Path(args.output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with open(out_dir / "prediction.json", "w", encoding="utf-8") as f:
                            json.dump(pred_record, f, indent=2, ensure_ascii=False)
                        with open(out_dir / "predictions.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.warning(f"Text generation failed: {e}")
                        generated_response = ""

                # Compose analysis text
                if args.emit_text and args.analyze_with_response and generated_response:
                    analysis_text = prompt_text + " " + generated_response
                else:
                    analysis_text = prompt_text


                # Optional residual patching
                patch_handle = None
                if args.patch_layer is not None:
                    if any(v is None for v in [args.patch_t_lo, args.patch_t_hi]):
                        logger.error("--patch_layer requires --patch_t_lo and --patch_t_hi")
                        return
                    if donor_text is None:
                        logger.error("--patch_layer requires --donor_text or --donor_text_file")
                        return
                    try:
                        enc_d, _ = safe_tokenize_with_fallback(hf_tokenizer, donor_text, return_tensors="pt")
                        enc_d = {k: v.to(device) for k, v in enc_d.items()}
                        with torch.no_grad():
                            out_d = hf_model(**enc_d, output_hidden_states=True, output_attentions=False)
                        if len(out_d.hidden_states) <= args.patch_layer:
                            logger.error(f"Model only has {len(out_d.hidden_states)} layers, cannot patch layer {args.patch_layer}")
                            return
                        donor_hidden = out_d.hidden_states[args.patch_layer]
                        block = getattr(hf_model, "model", hf_model).layers[args.patch_layer]
                        patch_handle = block.register_forward_hook(
                            PatchResidualAt(args.patch_t_lo, args.patch_t_hi, donor_hidden)
                        )
                        logger.info(f"Patching layer {args.patch_layer} tokens {args.patch_t_lo}:{args.patch_t_hi}")
                    except Exception as e:
                        logger.error(f"Failed to setup patching: {e}")
                        return

                # Run GSP analysis
                try:
                    logger.info(f"Analyzing: '{analysis_text[:100]}...'")
                    results = framework.analyze_text(analysis_text)

                    if results and 'layer_diagnostics' in results:
                        framework.create_visualizations(results)

                        # Pretty print
                        print(f"\nGSP Analysis Results")
                        print(f"Model: {config.model_name}")
                        gold_str = gold if gold is not None else '—'
                        corr_str = (str(is_correct) if gold is not None else "N/A")
                        print(f"QID: {qid if qid else '—'} | Style: {getattr(args, 'style', '—')}")
                        print(f"Prediction: {prediction} (Gold: {gold_str}, Correct: {corr_str})")
                        shown = analysis_text.replace("\n", " ")
                        print(f"Text: '{shown[:50]}...'")
                        print("=" * 80)
                        print(f"{'Layer':>5} {'Energy':>12} {'SMI':>8} {'Entropy':>8} {'HFER':>8} {'Fiedler':>8}")
                        print("-" * 80)
                        for i, diag in enumerate(results['layer_diagnostics']):
                            print(f"{i:5d} {diag.energy:12.4f} {diag.smoothness_index:8.4f} "
                                  f"{diag.spectral_entropy:8.4f} {diag.hfer:8.4f} {diag.fiedler_value:8.4f}")

                        energies = [d.energy for d in results['layer_diagnostics']]
                        print(f"\nSummary Statistics:")
                        print(f"Peak Energy: {max(energies):12.4f} (Layer {energies.index(max(energies))})")
                        print(f"Final Energy: {energies[-1]:11.4f} (Reduction: {max(energies)/energies[-1]:.1f}x)")

                        # ---- Append per-layer CSV if requested ----
                        try:
                            if getattr(args, "perlayer_out", None):
                                import csv
                                perlayer_path = Path(args.perlayer_out)
                                perlayer_path.parent.mkdir(parents=True, exist_ok=True)
                                has_header = perlayer_path.exists() and perlayer_path.stat().st_size > 0
                                with open(perlayer_path, 'a', newline='', encoding='utf-8') as f:
                                    w = csv.writer(f)
                                    if not has_header:
                                        w.writerow([
                                            "qid","model","style","layer",
                                            "fiedler_value","energy","smoothness_index","spectral_entropy","hfer",
                                            "prediction","gold","is_correct","source"
                                        ])
                                    for li, d in enumerate(results['layer_diagnostics']):
                                        w.writerow([
                                            qid or "",
                                            args.model,
                                            (args.style or ""),
                                            li,
                                            float(d.fiedler_value),
                                            float(d.energy),
                                            float(d.smoothness_index),
                                            float(d.spectral_entropy),
                                            float(d.hfer),
                                            prediction,
                                            gold,
                                            (bool(is_correct) if gold is not None else None),
                                            "cli_2"
                                        ])
                                logger.info(f"Appended per-layer rows to {perlayer_path}")
                        except Exception as e:
                            logger.warning(f"Failed to append per-layer CSV: {e}")
                    else:
                        logger.error("Analysis returned empty or invalid results")

                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    import traceback; traceback.print_exc()

                finally:
                    if patch_handle is not None:
                        patch_handle.remove()

            # ========= dispatch: single vs batch =========
            if single_text is not None:
                run_one(single_text, getattr(args, "qid", None), getattr(args, "gold", None))
            elif args.input_file:
                path = Path(args.input_file)
                if not path.exists():
                    logger.error(f"--input_file not found: {path}")
                    return
                logger.info(f"Batch mode from: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        qid = None; gold = None; text = line
                        # TSV format: qid<TAB>question<TAB>gold (gold optional)
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            qid, text = parts[0].strip(), parts[1].strip()
                        if len(parts) >= 3:
                            gold = parts[2].strip()
                        run_one(text, qid, gold)

            # cleanup ablation
            for h in ablation_handles:
                h.remove()

    except Exception as e:
        logger.error(f"Failed to analyze with model {args.model}: {e}")
        if "out of memory" in str(e).lower():
            logger.info("Try --load_in_8bit, --load_in_4bit, or --gradient_checkpointing")
        elif "tuple index out of range" in str(e).lower():
            logger.info("This may be a model compatibility issue.")
        raise

    logger.info(f"Analysis complete. Results saved to {args.output_dir}")


def cmd_compare(args):
    """Handle compare command with modern model support"""
    logger.info(f"Starting model comparison with {len(args.models)} models...")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    for model_name in args.models:
        logger.info(f"Analyzing with model: {model_name}")
        
        try:
            config = create_model_config(model_name, args)
            config.output_dir = str(output_path / model_name)
            config.verbose = False
            
            with GSPDiagnosticsFramework(config) as framework:
                framework.instrumenter.load_model(config.model_name)
                
                # Setup model
                hf_model = setup_model_for_analysis(framework.instrumenter.model)
                
                result = framework.analyze_text(args.text, save_results=False)
                
                if not result or 'layer_diagnostics' not in result:
                    logger.warning(f"No valid results for model {model_name}")
                    continue
                
                # Extract metrics with model info
                model_metrics = {
                    'model': model_name,
                    'full_name': config.model_name,
                    'max_length': config.max_length,
                    'layers': []
                }
                
                for i, diag in enumerate(result['layer_diagnostics']):
                    layer_metrics = {
                        'layer': int(i),
                        'energy': float(diag.energy),
                        'smoothness_index': float(diag.smoothness_index),
                        'spectral_entropy': float(diag.spectral_entropy),
                        'hfer': float(diag.hfer),
                        'fiedler_value': float(diag.fiedler_value)
                    }
                    model_metrics['layers'].append(layer_metrics)
                
                comparison_results[model_name] = model_metrics
                logger.info(f"Successfully analyzed {model_name} with {len(model_metrics['layers'])} layers")
                
        except Exception as e:
            logger.error(f"Failed to analyze with model {model_name}: {e}")
            continue
    
    if not comparison_results:
        logger.error("No successful model analyses completed")
        return
    
    # Save comparison results
    with open(output_path / "comparison_results.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, model_data in comparison_results.items():
        for layer_data in model_data['layers']:
            row = {
                'model': model_name,
                'full_name': model_data['full_name'],
                **layer_data
            }
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path / "comparison_data.csv", index=False)
    
    # Print enhanced summary comparison
    print(f"\nModel Comparison Results")
    print(f"Text: '{args.text[:50]}...'")
    print("=" * 100)
    
    # Show model info
    print("\nModels Analyzed:")
    for model_name, model_data in comparison_results.items():
        print(f"  {model_name:15} -> {model_data['full_name']} ({len(model_data['layers'])} layers)")
    
    if 'all' in args.metrics:
        metrics_to_show = ['energy', 'smoothness_index', 'spectral_entropy', 'hfer', 'fiedler_value']
    else:
        metrics_to_show = [m if m != 'fiedler' else 'fiedler_value' for m in args.metrics if m != 'all']
    
    # Show detailed comparison
    for metric in metrics_to_show:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        metric_summary = df.groupby('model')[metric].agg(['mean', 'std', 'min', 'max']).round(4)
        print(metric_summary)
    
    # Calculate model efficiency metrics
    print(f"\nModel Efficiency Analysis:")
    print("-" * 40)
    for model_name, model_data in comparison_results.items():
        layers = model_data['layers']
        energies = [l['energy'] for l in layers]
        peak_energy = max(energies)
        final_energy = energies[-1]
        reduction_ratio = peak_energy / final_energy if final_energy > 0 else float('inf')
        
        print(f"{model_name:15}: Peak={peak_energy:8.0f}, Final={final_energy:8.0f}, "
              f"Reduction={reduction_ratio:6.1f}x")
    
    # Create plots if requested
    if args.create_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style for better plots
            plt.style.use('default')
            
            for metric in metrics_to_show:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Box plot comparing models
                try:
                    sns.boxplot(data=df, x='model', y=metric, ax=ax1)
                    ax1.set_title(f'{metric.replace("_", " ").title()} Distribution by Model')
                    ax1.tick_params(axis='x', rotation=45)
                except:
                    # Fallback if seaborn fails
                    df.boxplot(column=metric, by='model', ax=ax1)
                    ax1.set_title(f'{metric.replace("_", " ").title()} Distribution by Model')
                
                # Line plot across layers
                for model in args.models:
                    if model in comparison_results:
                        model_data = df[df['model'] == model]
                        ax2.plot(model_data['layer'], model_data[metric], 
                                marker='o', label=model, linewidth=2, markersize=4)
                
                ax2.set_xlabel('Layer')
                ax2.set_ylabel(metric.replace('_', ' ').title())
                ax2.set_title(f'{metric.replace("_", " ").title()} Across Layers')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            logger.info(f"Comparison plots saved to {output_path}")
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")
    
    logger.info(f"Comparison complete. Results saved to {args.output_dir}")


def cmd_robustness(args):
    """Handle robustness command with modern model support"""
    logger.info(f"Starting robustness analysis with model: {args.model}")
    
    # Create model-aware configuration
    config = create_model_config(args.model, args)
    config.output_dir = args.output_dir
    config.verbose = True
    
    try:
        # For now, implement basic robustness analysis without external dependencies
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            
            # Setup model
            hf_model = setup_model_for_analysis(framework.instrumenter.model)
            
            # Analyze original text
            logger.info(f"Analyzing original text: '{args.text}'")
            original_result = framework.analyze_text(args.text, save_results=False)
            
            # Simple perturbation analysis - just show what we would do
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save original analysis
            results_summary = {
                'original_text': args.text,
                'model': config.model_name,
                'original_analysis': {
                    'layer_count': len(original_result['layer_diagnostics']),
                    'peak_energy': max(d.energy for d in original_result['layer_diagnostics']),
                    'final_energy': original_result['layer_diagnostics'][-1].energy,
                    'final_hfer': original_result['layer_diagnostics'][-1].hfer,
                    'final_entropy': original_result['layer_diagnostics'][-1].spectral_entropy
                }
            }
            
            # Save results
            with open(output_path / "robustness_results.json", "w") as f:
                json.dump(results_summary, f, indent=2)
            
            print("Robustness Analysis Complete!")
            print(f"Results saved to: {args.output_dir}")
            print(f"Model: {config.model_name}")
            print(f"Peak Energy: {results_summary['original_analysis']['peak_energy']:.2f}")
            print(f"Final Energy: {results_summary['original_analysis']['final_energy']:.2f}")
            
    except Exception as e:
        logger.error(f"Robustness analysis failed: {e}")
        raise


def cmd_evaluate(args):
    """Handle evaluate command with modern model support"""
    logger.info(f"Starting benchmark evaluation with model: {args.model}")
    
    # Create model-aware configuration
    config = create_model_config(args.model, args)
    config.output_dir = args.output_dir
    config.verbose = True
    
    try:
        # Simple evaluation implementation
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            
            # Setup model
            hf_model = setup_model_for_analysis(framework.instrumenter.model)
            
            # Sample evaluation texts
            test_texts = [
                "The capital of France is Paris.",
                "Water boils at 100 degrees Celsius.",
                "Shakespeare wrote Romeo and Juliet.",
                "The capital of France is London.",  # Error case
                "Two plus two equals seven."  # Logical error
            ]
            
            logger.info(f"Running evaluation on {len(test_texts)} test cases")
            
            results = []
            for i, text in enumerate(test_texts):
                logger.info(f"Evaluating text {i+1}/{len(test_texts)}: '{text[:50]}...'")
                
                try:
                    analysis = framework.analyze_text(text, save_results=False)
                    
                    result = {
                        'text_id': i,
                        'text': text,
                        'is_factual': i < 3,  # First 3 are factual
                        'peak_energy': max(d.energy for d in analysis['layer_diagnostics']),
                        'final_energy': analysis['layer_diagnostics'][-1].energy,
                        'final_hfer': analysis['layer_diagnostics'][-1].hfer,
                        'final_entropy': analysis['layer_diagnostics'][-1].spectral_entropy,
                        'energy_reduction': max(d.energy for d in analysis['layer_diagnostics']) / analysis['layer_diagnostics'][-1].energy
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate text {i}: {e}")
                    continue
            
            # Save results
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            evaluation_summary = {
                'model': config.model_name,
                'total_texts': len(test_texts),
                'successful_analyses': len(results),
                'results': results
            }
            
            with open(output_path / "evaluation_results.json", "w") as f:
                json.dump(evaluation_summary, f, indent=2, default=str)
            
            # Print summary
            print("Evaluation Complete!")
            print(f"Model: {config.model_name}")
            print(f"Texts analyzed: {len(results)}/{len(test_texts)}")
            print(f"Results saved to: {args.output_dir}")
            
            if results:
                factual_results = [r for r in results if r['is_factual']]
                error_results = [r for r in results if not r['is_factual']]
                
                if factual_results and error_results:
                    print(f"\nFactual vs Error Comparison:")
                    print(f"Factual avg final HFER: {sum(r['final_hfer'] for r in factual_results)/len(factual_results):.3f}")
                    print(f"Error avg final HFER: {sum(r['final_hfer'] for r in error_results)/len(error_results):.3f}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def cmd_batch(args):
    """Handle batch command with modern model support"""
    logger.info("Starting batch processing with modern models...")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {args.config}: {e}")
        return
    
    # Override model if specified
    if hasattr(args, 'model') and args.model:
        logger.info(f"Overriding config model with: {args.model}")
        model_info = get_model_info(args.model)
        config_dict.update({
            'model_name': model_info['name'],
            'max_length': model_info['max_length'],
            'torch_dtype': model_info.get('torch_dtype', 'float32'),
        })
    
    config = GSPConfig(**config_dict)
    config.output_dir = args.output_dir
    
    # Find input files
    input_path = Path(args.input_dir)
    files = list(input_path.glob(args.file_pattern))
    
    if not files:
        logger.error(f"No files found matching pattern '{args.file_pattern}' in {args.input_dir}")
        return
    
    logger.info(f"Found {len(files)} files to process with model: {config.model_name}")
    
    # Process files
    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            
            # Setup model
            hf_model = setup_model_for_analysis(framework.instrumenter.model)
            
            all_results = []
            processed_files = 0
            processed_texts = 0
            
            for file_path in files:
                logger.info(f"Processing {file_path.name}...")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts = [line.strip() for line in f if line.strip()]
                    
                    file_results = 0
                    
                    for j, text in enumerate(texts):
                        try:
                            result = framework.analyze_text(text, save_results=False)
                            
                            # Extract summary metrics
                            if result and 'layer_diagnostics' in result:
                                for layer_idx, diag in enumerate(result['layer_diagnostics']):
                                    all_results.append({
                                        'file': file_path.name,
                                        'text_id': j,
                                        'text': text[:200],  # Truncate for storage
                                        'text_length': len(text),
                                        'layer': layer_idx,
                                        'energy': float(diag.energy),
                                        'smoothness_index': float(diag.smoothness_index),
                                        'spectral_entropy': float(diag.spectral_entropy),
                                        'hfer': float(diag.hfer),
                                        'fiedler_value': float(diag.fiedler_value)
                                    })
                                
                                file_results += 1
                                processed_texts += 1
                                
                                if processed_texts % 10 == 0:
                                    logger.info(f"Processed {processed_texts} texts...")
                            else:
                                logger.warning(f"No valid results for text {j} in {file_path.name}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to process text {j} in {file_path.name}: {e}")
                            continue
                    
                    processed_files += 1
                    logger.info(f"Completed {file_path.name}: {file_results} texts processed")
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file_path.name}: {e}")
                    continue
        
        # Save batch results
        if all_results:
            df = pd.DataFrame(all_results)
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path / "batch_results.csv", index=False)
            
            print(f"\nBatch Processing Complete!")
            print(f"Model: {config.model_name}")
            print(f"Files processed: {processed_files}/{len(files)}")
            print(f"Texts processed: {processed_texts}")
            print(f"Results saved to: {args.output_dir}")
            
        else:
            logger.error("No results generated from batch processing")
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


def main():
    """Main CLI function with enhanced modern model support and error handling"""
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print(f"\nQuick start with modern models:")
        print(f"  python cli.py analyze --model llama-3.2-1b --text \"Your text here\"")
        print(f"  python cli.py list-models")
        sys.exit(1)
    
    # Check for required packages
    try:
        import torch
        logger.info(f"PyTorch {torch.__version__} loaded")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
        else:
            logger.info("Using CPU")
    except ImportError:
        logger.error("PyTorch is required but not installed. Please install torch.")
        sys.exit(1)
    
    # Route to appropriate command handler
    try:
        if args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'robustness':
            cmd_robustness(args)
        elif args.command == 'evaluate':
            cmd_evaluate(args)
        elif args.command == 'batch':
            cmd_batch(args)
        elif args.command == 'compare':
            cmd_compare(args)
        elif args.command == 'list-models':
            cmd_list_models(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        
        # Provide helpful suggestions for common errors
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda out of memory" in error_msg:
            print("\nMemory optimization suggestions:")
            print("  --load_in_8bit          # Use 8-bit quantization")
            print("  --load_in_4bit          # Use 4-bit quantization") 
            print("  --gradient_checkpointing # Trade compute for memory")
            print("  --precision float16     # Use half precision")
        elif "model not found" in error_msg or "repository not found" in error_msg:
            print("\nModel loading suggestions:")
            print("  python cli.py list-models  # See available shortcuts")
            print("  Check model name spelling and availability on HuggingFace")
        elif "permission" in error_msg or "authentication" in error_msg:
            print("\nModel access suggestions:")
            print("  huggingface-cli login   # Login to access gated models")
            print("  Check if model requires special permissions")
        elif "tuple index out of range" in error_msg:
            print("\nCompatibility suggestions:")
            print("  This error often occurs with model output parsing")
            print("  Try using a different model or update your GSP framework")
            print("  Check that the model architecture is supported")
        elif "tokenization" in error_msg:
            print("\nTokenization suggestions:")
            print("  Check input text encoding (should be UTF-8)")
            print("  Try with a different input text")
            print("  Some models have specific tokenization requirements")
        
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
            
        sys.exit(1)


if __name__ == "__main__":
    main()