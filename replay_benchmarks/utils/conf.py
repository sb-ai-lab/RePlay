import os
import yaml
from typing import Dict


def load_yaml(file_path: str) -> Dict:
    """Load a single YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict, update: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(base_config_path: str, config_dir: str) -> Dict:
    """Load base configuration and merge sub-configurations."""
    base_config = load_yaml(base_config_path)
    defaults = base_config.get("defaults", [])

    for default in defaults:
        for key, subconfig in default.items():
            if subconfig is None:
                continue
            subconfig_path = os.path.join(config_dir, key, f"{subconfig}.yaml")
            if not os.path.exists(subconfig_path):
                raise FileNotFoundError(f"Configuration file not found: {subconfig_path}")
            subconfig_data = load_yaml(subconfig_path)
            base_config = merge_dicts(base_config, subconfig_data)

    base_config.pop("defaults", None)
    return base_config