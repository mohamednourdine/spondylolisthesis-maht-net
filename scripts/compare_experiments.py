#!/usr/bin/env python3
"""
Compare training results across different experiments.

Usage:
    python scripts/compare_experiments.py --model unet
    python scripts/compare_experiments.py --model unet --top 5
    python scripts/compare_experiments.py --list-all
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_experiment_results(experiment_dir: Path) -> Dict[str, Any]:
    """Load results from an experiment directory."""
    history_file = experiment_dir / 'training_history.json'
    config_file = experiment_dir / 'config.json'
    
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return {
        'name': experiment_dir.name,
        'path': str(experiment_dir),
        'best_val_loss': history.get('best_val_loss', float('inf')),
        'final_train_loss': history['train_losses'][-1] if history.get('train_losses') else None,
        'final_val_loss': history['val_losses'][-1] if history.get('val_losses') else None,
        'num_epochs': len(history.get('train_losses', [])),
        'batch_size': config.get('batch_size', 'N/A'),
        'learning_rate': config.get('learning_rate', 'N/A'),
        'val_metrics': history.get('val_metrics', []),
    }


def list_experiments(model_name: str = None, results_dir: Path = None) -> List[Dict[str, Any]]:
    """List all experiments, optionally filtered by model."""
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'experiments' / 'results'
    
    experiments = []
    
    if model_name:
        # List experiments for specific model
        model_dir = results_dir / model_name
        if not model_dir.exists():
            print(f"No experiments found for model: {model_name}")
            return []
        
        for exp_dir in sorted(model_dir.iterdir()):
            if exp_dir.is_dir():
                result = load_experiment_results(exp_dir)
                if result:
                    result['model'] = model_name
                    experiments.append(result)
    else:
        # List all experiments across all models
        for model_dir in sorted(results_dir.iterdir()):
            if model_dir.is_dir() and model_dir.name != '.git':
                for exp_dir in sorted(model_dir.iterdir()):
                    if exp_dir.is_dir():
                        result = load_experiment_results(exp_dir)
                        if result:
                            result['model'] = model_dir.name
                            experiments.append(result)
    
    return experiments


def print_experiment_table(experiments: List[Dict[str, Any]], top_n: int = None):
    """Print experiments in a formatted table."""
    if not experiments:
        print("No experiments found.")
        return
    
    # Sort by best validation loss
    experiments = sorted(experiments, key=lambda x: x['best_val_loss'])
    
    if top_n:
        experiments = experiments[:top_n]
    
    # Print header
    print("\n" + "="*120)
    print(f"{'Rank':<6}{'Model':<15}{'Experiment':<40}{'Epochs':<8}{'Best Val Loss':<15}{'LR':<10}{'Batch':<8}")
    print("="*120)
    
    # Print experiments
    for rank, exp in enumerate(experiments, 1):
        model = exp['model']
        name = exp['name'][:38] + '..' if len(exp['name']) > 38 else exp['name']
        epochs = exp['num_epochs']
        best_loss = f"{exp['best_val_loss']:.4f}"
        lr = f"{exp['learning_rate']}" if isinstance(exp['learning_rate'], str) else f"{exp['learning_rate']:.0e}"
        batch = str(exp['batch_size'])
        
        print(f"{rank:<6}{model:<15}{name:<40}{epochs:<8}{best_loss:<15}{lr:<10}{batch:<8}")
    
    print("="*120 + "\n")


def print_detailed_metrics(experiment: Dict[str, Any]):
    """Print detailed metrics for an experiment."""
    print("\n" + "="*80)
    print(f"Experiment: {experiment['name']}")
    print("="*80)
    print(f"Model:           {experiment['model']}")
    print(f"Path:            {experiment['path']}")
    print(f"Epochs:          {experiment['num_epochs']}")
    print(f"Batch Size:      {experiment['batch_size']}")
    print(f"Learning Rate:   {experiment['learning_rate']}")
    print(f"\nTraining Results:")
    print(f"  Best Val Loss:   {experiment['best_val_loss']:.4f}")
    print(f"  Final Train Loss: {experiment['final_train_loss']:.4f}")
    print(f"  Final Val Loss:   {experiment['final_val_loss']:.4f}")
    
    # Print final epoch metrics if available
    if experiment['val_metrics']:
        last_metrics = experiment['val_metrics'][-1]
        print(f"\nFinal Validation Metrics:")
        for key, value in last_metrics.items():
            if key != 'loss':
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--model', type=str, default=None,
                       help='Filter by model name (unet, maht-net, etc.)')
    parser.add_argument('--top', type=int, default=None,
                       help='Show only top N experiments')
    parser.add_argument('--list-all', action='store_true',
                       help='List all experiments across all models')
    parser.add_argument('--details', type=str, default=None,
                       help='Show detailed metrics for specific experiment (use experiment name)')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list_all:
        experiments = list_experiments()
    else:
        experiments = list_experiments(model_name=args.model)
    
    if not experiments:
        print("No experiments found.")
        return
    
    # Show details for specific experiment
    if args.details:
        matching = [e for e in experiments if args.details in e['name']]
        if matching:
            print_detailed_metrics(matching[0])
        else:
            print(f"No experiment found matching: {args.details}")
        return
    
    # Print table
    print_experiment_table(experiments, top_n=args.top)
    
    # Print summary
    if args.model:
        print(f"Total experiments for {args.model}: {len(experiments)}")
    else:
        print(f"Total experiments across all models: {len(experiments)}")
        model_counts = {}
        for exp in experiments:
            model_counts[exp['model']] = model_counts.get(exp['model'], 0) + 1
        print("\nExperiments per model:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count}")


if __name__ == '__main__':
    main()
