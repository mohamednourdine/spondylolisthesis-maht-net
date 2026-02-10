"""
Hyperparameter Tuning Config Loader

Utilities for loading and managing MAHT-Net hyperparameter configurations.
Supports:
- YAML config loading with inheritance (_base_ field)
- Experiment variant generation
- Config merging and overriding
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy
import json


CONFIG_DIR = Path(__file__).parent.parent / "experiments" / "configs" / "tuning"


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    Override values take precedence over base values.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(config_name: str, config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """
    Load a config file with inheritance support.
    
    If the config contains a '_base_' field, load and merge with base config.
    
    Args:
        config_name: Config filename (with or without .yaml extension)
        config_dir: Directory containing config files
        
    Returns:
        Loaded and merged configuration dictionary
    """
    if not config_name.endswith('.yaml'):
        config_name = config_name + '.yaml'
    
    config_path = config_dir / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_yaml(config_path)
    
    # Handle inheritance
    if '_base_' in config:
        base_name = config.pop('_base_')
        base_config = load_config(base_name, config_dir)
        config = deep_merge(base_config, config)
    
    return config


def get_experiment_config(
    base_config: str,
    experiment_name: str,
    config_dir: Path = CONFIG_DIR
) -> Dict[str, Any]:
    """
    Get a specific experiment configuration.
    
    Loads the base config and applies experiment-specific overrides.
    
    Args:
        base_config: Base config filename
        experiment_name: Name of the experiment in 'experiments' list
        config_dir: Directory containing config files
        
    Returns:
        Complete configuration for the experiment
    """
    config = load_config(base_config, config_dir)
    
    # Find the experiment
    experiments = config.get('experiments', [])
    experiment = None
    
    for exp in experiments:
        if exp.get('name') == experiment_name:
            experiment = exp
            break
    
    if experiment is None:
        # Check other sections (phase_ablation, vam_ablation, etc.)
        for section_name in ['phase_ablation', 'vam_ablation', 'component_ablation', 
                             'loss_ablation', 'augmentation_ablation']:
            section = config.get(section_name, [])
            for exp in section:
                if exp.get('name') == experiment_name:
                    experiment = exp
                    break
            if experiment:
                break
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in {base_config}")
    
    # Remove metadata fields before merging
    exp_config = {k: v for k, v in experiment.items() 
                  if k not in ['name', 'description', 'expected_delta']}
    
    # Merge with base (excluding experiments list)
    base_without_experiments = {k: v for k, v in config.items() 
                                if k not in ['experiments', 'phase_ablation', 'vam_ablation',
                                             'component_ablation', 'loss_ablation', 
                                             'augmentation_ablation', 'recommended']}
    
    return deep_merge(base_without_experiments, exp_config)


def list_experiments(config_name: str, config_dir: Path = CONFIG_DIR) -> List[Dict[str, str]]:
    """
    List all experiments in a config file.
    
    Args:
        config_name: Config filename
        config_dir: Directory containing config files
        
    Returns:
        List of experiment info dicts with name and description
    """
    config = load_config(config_name, config_dir)
    
    experiments = []
    
    # Main experiments list
    for exp in config.get('experiments', []):
        experiments.append({
            'name': exp.get('name', 'unnamed'),
            'description': exp.get('description', ''),
            'section': 'experiments'
        })
    
    # Ablation study sections
    for section_name in ['phase_ablation', 'vam_ablation', 'component_ablation', 
                         'loss_ablation', 'augmentation_ablation']:
        for exp in config.get(section_name, []):
            experiments.append({
                'name': exp.get('name', 'unnamed'),
                'description': exp.get('description', ''),
                'section': section_name,
                'expected_delta': exp.get('expected_delta', '')
            })
    
    # Recommended config
    if 'recommended' in config:
        experiments.append({
            'name': config['recommended'].get('name', 'recommended'),
            'description': config['recommended'].get('description', 'Recommended configuration'),
            'section': 'recommended'
        })
    
    return experiments


def generate_all_configs(
    tuning_file: str,
    output_dir: Optional[Path] = None,
    config_dir: Path = CONFIG_DIR
) -> List[Path]:
    """
    Generate individual config files for all experiments in a tuning file.
    
    Args:
        tuning_file: Tuning config filename (e.g., 'lr_optimizer_tuning.yaml')
        output_dir: Directory to save generated configs (default: config_dir/generated)
        config_dir: Directory containing tuning configs
        
    Returns:
        List of paths to generated config files
    """
    if output_dir is None:
        output_dir = config_dir / "generated"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = list_experiments(tuning_file, config_dir)
    generated_files = []
    
    for exp in experiments:
        try:
            config = get_experiment_config(tuning_file, exp['name'], config_dir)
            
            # Add experiment metadata
            config['experiment'] = {
                'name': exp['name'],
                'description': exp.get('description', ''),
                'source_file': tuning_file,
                'section': exp.get('section', 'experiments')
            }
            
            # Save to file
            output_path = output_dir / f"{exp['name']}.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(output_path)
            print(f"  ✓ Generated: {output_path.name}")
            
        except Exception as e:
            print(f"  ✗ Failed to generate {exp['name']}: {e}")
    
    return generated_files


def print_config_summary(config: Dict[str, Any], indent: int = 0) -> None:
    """Print a formatted summary of a config."""
    prefix = "  " * indent
    
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config_summary(value, indent + 1)
        elif isinstance(value, list) and len(value) > 3:
            print(f"{prefix}{key}: [{len(value)} items]")
        else:
            print(f"{prefix}{key}: {value}")


class TuningConfig:
    """
    Hyperparameter tuning configuration manager.
    
    Usage:
        # Load base config
        cfg = TuningConfig('maht_net_base')
        
        # Access parameters
        lr = cfg.optimizer.lr_backbone
        
        # Load specific experiment
        cfg = TuningConfig.from_experiment('lr_optimizer_tuning', 'lr_sweep_1e3')
        
        # Get as dict for model initialization
        model_cfg = cfg.get_model_config()
    """
    
    def __init__(self, config_name: str, config_dir: Path = CONFIG_DIR):
        """Load a configuration file."""
        self._config = load_config(config_name, config_dir)
        self._config_name = config_name
    
    @classmethod
    def from_experiment(
        cls,
        tuning_file: str,
        experiment_name: str,
        config_dir: Path = CONFIG_DIR
    ) -> 'TuningConfig':
        """Create a TuningConfig from a specific experiment."""
        config = get_experiment_config(tuning_file, experiment_name, config_dir)
        
        instance = cls.__new__(cls)
        instance._config = config
        instance._config_name = f"{tuning_file}/{experiment_name}"
        
        return instance
    
    def __getattr__(self, name: str) -> Any:
        """Access config sections as attributes."""
        if name.startswith('_'):
            raise AttributeError(name)
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return deepcopy(self._config)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Extract model-related configuration for model initialization."""
        return self._config.get('model', {})
    
    def get_loss_config(self) -> Dict[str, Any]:
        """Extract loss-related configuration."""
        return self._config.get('loss', {})
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Extract optimizer configuration."""
        return self._config.get('optimizer', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Extract training configuration."""
        return self._config.get('training', {})
    
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"TuningConfig({self._config_name})"


class ConfigSection:
    """Wrapper for config sections to allow attribute access."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"Config section has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._data)
    
    def __repr__(self) -> str:
        return f"ConfigSection({list(self._data.keys())})"


# ==============================================================================
# CLI for config management
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAHT-Net Hyperparameter Config Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List experiments in a config')
    list_parser.add_argument('config', help='Config filename')
    
    # Show experiment
    show_parser = subparsers.add_parser('show', help='Show experiment configuration')
    show_parser.add_argument('config', help='Config filename')
    show_parser.add_argument('experiment', help='Experiment name')
    
    # Generate configs
    gen_parser = subparsers.add_parser('generate', help='Generate individual config files')
    gen_parser.add_argument('config', help='Config filename')
    gen_parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        print(f"\nExperiments in {args.config}:")
        print("=" * 60)
        experiments = list_experiments(args.config)
        for exp in experiments:
            delta = f" ({exp['expected_delta']})" if exp.get('expected_delta') else ""
            print(f"  [{exp['section']}] {exp['name']}: {exp['description']}{delta}")
        print(f"\nTotal: {len(experiments)} experiments")
        
    elif args.command == 'show':
        config = get_experiment_config(args.config, args.experiment)
        print(f"\nConfiguration: {args.config}/{args.experiment}")
        print("=" * 60)
        print_config_summary(config)
        
    elif args.command == 'generate':
        output_dir = Path(args.output) if args.output else None
        print(f"\nGenerating configs from {args.config}...")
        files = generate_all_configs(args.config, output_dir)
        print(f"\nGenerated {len(files)} config files")
        
    else:
        parser.print_help()
