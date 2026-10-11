"""
Configuration Loader and Validator
Loads, validates, and provides typed access to experiment and pipeline configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigValidationError(Exception):
    """Raised when configuration values fail validation rules."""
    pass


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML file and returns its dictionary representation."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


@dataclass
class AppConfig:
    """Master configuration structure holding all component configurations."""
    experiment_id: str
    description: str
    seed: int
    data: Dict[str, Any]
    features: Dict[str, Any]
    labels: Dict[str, Any]
    validation: Dict[str, Any]
    models: Dict[str, Any]
    portfolio: Dict[str, Any]
    artifacts: Dict[str, Any]
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Enforces critical configuration sanity checks."""
        # 1. Label horizon check
        horizon = self.labels.get("returns", {}).get("horizon_bars", 0)
        if horizon <= 0:
            raise ConfigValidationError(f"Return horizon must be positive, got {horizon}")

        vol_horizon = self.labels.get("volatility", {}).get("horizon_bars", 0)
        if vol_horizon <= 0:
            raise ConfigValidationError(f"Volatility horizon must be positive, got {vol_horizon}")

        # 2. Sequence length check
        seq_len = self.features.get("sequence", {}).get("sequence_length", 0)
        if seq_len < 3:
            raise ConfigValidationError(f"Sequence length must be at least 3 bars, got {seq_len}")

        # 3. Transformer dimension compatibility
        tf_cfg = self.models.get("transformer", {})
        d_model = tf_cfg.get("d_model", 64)
        nhead = tf_cfg.get("nhead", 4)
        if d_model % nhead != 0:
            raise ConfigValidationError(
                f"Transformer d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        # 4. Decile universe constraints
        num_quantiles = self.portfolio.get("portfolio", {}).get("num_quantiles", 10)
        min_universe = self.portfolio.get("portfolio", {}).get("min_universe_size", 10)
        if min_universe < num_quantiles:
            raise ConfigValidationError(
                f"min_universe_size ({min_universe}) cannot be smaller than num_quantiles ({num_quantiles})"
            )

        # 5. Holdout test ratio
        holdout = self.validation.get("protocol", {}).get("holdout_test_ratio", 0.2)
        if not (0.0 < holdout < 0.5):
            raise ConfigValidationError(
                f"Holdout test ratio must be between 0.0 and 0.5, got {holdout}"
            )


def load_master_config(experiment_yaml_path: str | Path = "configs/experiment.yaml") -> AppConfig:
    """
    Loads experiment.yaml and all linked sub-configs, performs validation, and returns AppConfig.
    """
    base_dir = Path(experiment_yaml_path).parent.parent
    exp_dict = load_yaml(experiment_yaml_path)

    data_path = exp_dict.get("data_config", "configs/data.yaml")
    feat_path = exp_dict.get("features_config", "configs/features.yaml")
    lbl_path = exp_dict.get("labels_config", "configs/labels.yaml")
    val_path = exp_dict.get("validation_config", "configs/validation.yaml")
    mod_path = exp_dict.get("models_config", "configs/models.yaml")
    port_path = exp_dict.get("portfolio_config", "configs/portfolio.yaml")

    data_cfg = load_yaml(base_dir / data_path if not Path(data_path).is_absolute() else data_path)
    feat_cfg = load_yaml(base_dir / feat_path if not Path(feat_path).is_absolute() else feat_path)
    lbl_cfg = load_yaml(base_dir / lbl_path if not Path(lbl_path).is_absolute() else lbl_path)
    val_cfg = load_yaml(base_dir / val_path if not Path(val_path).is_absolute() else val_path)
    mod_cfg = load_yaml(base_dir / mod_path if not Path(mod_path).is_absolute() else mod_path)
    port_cfg = load_yaml(base_dir / port_path if not Path(port_path).is_absolute() else port_path)

    config = AppConfig(
        experiment_id=exp_dict.get("experiment_id", "default_experiment"),
        description=exp_dict.get("description", ""),
        seed=exp_dict.get("seed", 42),
        data=data_cfg,
        features=feat_cfg,
        labels=lbl_cfg,
        validation=val_cfg,
        models=mod_cfg,
        portfolio=port_cfg,
        artifacts=exp_dict.get("artifacts", {}),
        raw_dict=exp_dict,
    )
    config.validate()
    return config
