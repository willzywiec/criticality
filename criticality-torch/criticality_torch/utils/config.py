"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {config_path}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "data": {
            "code": "mcnp",
            "test_split": 0.2,
            "mass_threshold": 200.0,
        },
        "model": {
            "layers": "8192-256-256-256-256-16",
            "activation": "relu",
        },
        "training": {
            "ensemble_size": 5,
            "epochs": 1500,
            "batch_size": 8192,
            "learning_rate": 0.00075,
            "optimizer": "adamax",
            "loss": "sse",
            "val_split": 0.2,
        },
        "bayesian_network": {
            "distribution": "gamma",
        },
        "risk": {
            "sample_size": 1000000,
            "risk_pool": 100,
            "usl": 0.95,
            "keff_cutoff": 0.9,
            "mass_cutoff": 100.0,
            "rad_cutoff": 7.0,
        },
        "output": {
            "base_dir": "./output",
            "save_checkpoints": True,
            "save_plots": True,
        }
    }
