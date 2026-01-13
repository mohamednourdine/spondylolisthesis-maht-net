#!/usr/bin/env python
"""
Clean up old experiment results to free disk space.

Features:
- List all experiments with size, date, and metrics
- Delete experiments older than N days
- Keep only top N best experiments per model
- Dry-run mode to preview deletions
- Interactive confirmation

Usage:
    # List all experiments
    python scripts/cleanup_experiments.py --list
    
    # Delete experiments older than 7 days (dry run)
    python scripts/cleanup_experiments.py --older-than 7 --dry-run
    
    # Keep only top 3 best experiments per model
    python scripts/cleanup_experiments.py --keep-best 3
    
    # Delete specific experiment
    python scripts/cleanup_experiments.py --delete "mac_512px_20260110_143022"
    
    # Force delete without confirmation
    python scripts/cleanup_experiments.py --older-than 30 --force
"""

import sys
import os
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_experiment_info(exp_path: Path) -> Dict:
    """Extract information about an experiment."""
    info = {
        'name': exp_path.name,
        'path': exp_path,
        'model': exp_path.parent.name,
        'size_mb': 0,
        'created': None,
        'best_mre': float('inf'),
        'best_sdr': 0,
        'epochs': 0,
        'has_best_model': False,
    }
    
    # Calculate size
    total_size = 0
    for f in exp_path.rglob('*'):
        if f.is_file():
            total_size += f.stat().st_size
    info['size_mb'] = total_size / (1024 * 1024)
    
    # Get creation time from directory
    try:
        info['created'] = datetime.fromtimestamp(exp_path.stat().st_mtime)
    except:
        pass
    
    # Try to parse timestamp from name (format: name_YYYYMMDD_HHMMSS)
    try:
        parts = exp_path.name.split('_')
        if len(parts) >= 2:
            date_str = parts[-2]
            time_str = parts[-1]
            if len(date_str) == 8 and len(time_str) == 6:
                info['created'] = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except:
        pass
    
    # Check for best model
    best_model = exp_path / 'best_model.pth'
    info['has_best_model'] = best_model.exists()
    
    # Try to load training history for metrics
    history_file = exp_path / 'training_history.json'
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Get best validation metrics
            if 'val_metrics' in history:
                for metrics in history['val_metrics']:
                    if 'MRE_px' in metrics:
                        info['best_mre'] = min(info['best_mre'], metrics['MRE_px'])
                    if 'SDR_24px' in metrics:
                        info['best_sdr'] = max(info['best_sdr'], metrics['SDR_24px'])
                info['epochs'] = len(history['val_metrics'])
        except:
            pass
    
    # Also try config.json for epoch count
    config_file = exp_path / 'config.json'
    if config_file.exists() and info['epochs'] == 0:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            if 'training' in config and 'num_epochs' in config['training']:
                info['epochs'] = config['training']['num_epochs']
        except:
            pass
    
    return info


def list_experiments(results_dir: Path, model: Optional[str] = None) -> List[Dict]:
    """List all experiments with their info."""
    experiments = []
    
    if not results_dir.exists():
        return experiments
    
    # Find model directories
    model_dirs = []
    if model:
        model_path = results_dir / model
        if model_path.exists():
            model_dirs = [model_path]
    else:
        model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    # Find experiments in each model directory
    for model_dir in model_dirs:
        for exp_dir in model_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                info = get_experiment_info(exp_dir)
                experiments.append(info)
    
    # Sort by creation date (newest first)
    experiments.sort(key=lambda x: x['created'] or datetime.min, reverse=True)
    
    return experiments


def format_size(size_mb: float) -> str:
    """Format size in human-readable form."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    elif size_mb >= 1:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb * 1024:.0f} KB"


def format_date(dt: Optional[datetime]) -> str:
    """Format datetime for display."""
    if dt is None:
        return "Unknown"
    
    now = datetime.now()
    diff = now - dt
    
    if diff.days == 0:
        return f"Today {dt.strftime('%H:%M')}"
    elif diff.days == 1:
        return f"Yesterday {dt.strftime('%H:%M')}"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    else:
        return dt.strftime("%Y-%m-%d")


def print_experiments(experiments: List[Dict], verbose: bool = False):
    """Print experiment list in a nice table."""
    if not experiments:
        print("No experiments found.")
        return
    
    # Calculate total size
    total_size = sum(e['size_mb'] for e in experiments)
    
    print(f"\n{'='*80}")
    print(f"Found {len(experiments)} experiments (Total: {format_size(total_size)})")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Model':<12} {'Experiment':<35} {'Size':>8} {'Date':<15} {'MRE':>8} {'SDR@24':>8}")
    print("-" * 90)
    
    for exp in experiments:
        model = exp['model'][:11]
        name = exp['name'][:34]
        size = format_size(exp['size_mb'])
        date = format_date(exp['created'])
        
        mre = f"{exp['best_mre']:.1f}px" if exp['best_mre'] < float('inf') else "N/A"
        sdr = f"{exp['best_sdr']*100:.1f}%" if exp['best_sdr'] > 0 else "N/A"
        
        # Mark best model with star
        marker = "★" if exp['has_best_model'] else " "
        
        print(f"{model:<12} {marker}{name:<34} {size:>8} {date:<15} {mre:>8} {sdr:>8}")
    
    print("-" * 90)
    print(f"{'Total':<48} {format_size(total_size):>8}")
    print(f"\n★ = has best_model.pth")


def delete_experiment(exp: Dict, dry_run: bool = True) -> Tuple[bool, str]:
    """Delete an experiment directory."""
    path = exp['path']
    
    if dry_run:
        return True, f"[DRY RUN] Would delete: {path}"
    
    try:
        shutil.rmtree(path)
        return True, f"Deleted: {path}"
    except Exception as e:
        return False, f"Failed to delete {path}: {e}"


def filter_old_experiments(experiments: List[Dict], days: int) -> List[Dict]:
    """Filter experiments older than N days."""
    cutoff = datetime.now() - timedelta(days=days)
    return [e for e in experiments if e['created'] and e['created'] < cutoff]


def get_experiments_to_keep(experiments: List[Dict], keep_best: int) -> List[Dict]:
    """Get experiments to delete, keeping top N best per model."""
    to_delete = []
    
    # Group by model
    by_model = {}
    for exp in experiments:
        model = exp['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(exp)
    
    # For each model, keep only the best N (by MRE, lower is better)
    for model, model_exps in by_model.items():
        # Sort by best MRE (ascending)
        sorted_exps = sorted(model_exps, key=lambda x: x['best_mre'])
        
        # Mark experiments beyond top N for deletion
        for i, exp in enumerate(sorted_exps):
            if i >= keep_best:
                to_delete.append(exp)
    
    return to_delete


def main():
    parser = argparse.ArgumentParser(
        description='Clean up old experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all experiments')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Filter by model name (e.g., unet)')
    parser.add_argument('--older-than', type=int, default=None,
                        help='Delete experiments older than N days')
    parser.add_argument('--keep-best', type=int, default=None,
                        help='Keep only top N best experiments per model')
    parser.add_argument('--delete', type=str, default=None,
                        help='Delete specific experiment by name')
    parser.add_argument('--all', action='store_true',
                        help='Delete ALL experiments (use with caution!)')
    parser.add_argument('--include-outputs', action='store_true',
                        help='Also delete test_evaluation/ and visualizations/ folders')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--results-dir', type=str, 
                        default=str(PROJECT_ROOT / 'experiments' / 'results'),
                        help='Path to results directory')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1
    
    # Get all experiments
    experiments = list_experiments(results_dir, args.model)
    
    # List mode
    if args.list or (not args.older_than and not args.keep_best and not args.delete and not args.all):
        print_experiments(experiments, verbose=True)
        return 0
    
    # Determine what to delete
    to_delete = []
    
    if args.all:
        # Delete ALL experiments
        to_delete = experiments
        # Don't return early - we may still want to delete output folders
    
    elif args.delete:
        # Delete specific experiment
        matches = [e for e in experiments if args.delete in e['name']]
        if not matches:
            print(f"No experiment found matching: {args.delete}")
            return 1
        to_delete = matches
        
    elif args.older_than:
        # Delete old experiments
        to_delete = filter_old_experiments(experiments, args.older_than)
        if not to_delete:
            print(f"No experiments older than {args.older_than} days found.")
            return 0
            
    elif args.keep_best:
        # Keep only best N
        to_delete = get_experiments_to_keep(experiments, args.keep_best)
        if not to_delete:
            print(f"Nothing to delete (already have ≤{args.keep_best} experiments per model).")
            return 0
    
    # Show what will be deleted
    if to_delete:
        print(f"\n{'='*60}")
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Experiments to delete: {len(to_delete)}")
        print(f"{'='*60}\n")
        
        total_size = 0
        for exp in to_delete:
            size = format_size(exp['size_mb'])
            date = format_date(exp['created'])
            print(f"  {exp['model']}/{exp['name']} ({size}, {date})")
            total_size += exp['size_mb']
        
        print(f"\nTotal space to free: {format_size(total_size)}")
        
        # Confirm deletion
        if not args.dry_run and not args.force:
            try:
                response = input(f"\nDelete {len(to_delete)} experiments? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return 0
            except KeyboardInterrupt:
                print("\nAborted.")
                return 0
        
        # Perform deletion
        success_count = 0
        for exp in to_delete:
            success, msg = delete_experiment(exp, dry_run=args.dry_run)
            print(msg)
            if success:
                success_count += 1
        
        print(f"\n{'Would delete' if args.dry_run else 'Deleted'}: {success_count}/{len(to_delete)} experiments")
        if not args.dry_run:
            print(f"Freed: {format_size(total_size)}")
    else:
        print("No experiments to delete.")
    
    # Handle additional output folders
    if args.include_outputs or args.all:
        experiments_dir = results_dir.parent  # experiments/
        output_folders = ['test_evaluation', 'visualizations']
        
        print(f"\n{'='*60}")
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Additional output folders:")
        print(f"{'='*60}\n")
        
        for folder_name in output_folders:
            folder_path = experiments_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                # Calculate folder size
                folder_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
                folder_size_mb = folder_size / (1024 * 1024)
                
                if args.dry_run:
                    print(f"  [DRY RUN] Would delete: {folder_path} ({format_size(folder_size_mb)})")
                else:
                    try:
                        shutil.rmtree(folder_path)
                        print(f"  Deleted: {folder_path} ({format_size(folder_size_mb)})")
                    except Exception as e:
                        print(f"  Failed to delete {folder_path}: {e}")
            else:
                print(f"  {folder_name}/ not found or empty")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
