"""
Central Experiment Runner
Orchestrates purged walk-forward cross validation, model fitting, out-of-sample prediction stitching,
and experiment manifest tracking.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import AppConfig
from ..features.pipeline import FeaturePipeline
from ..models.inference import UnifiedPredictor
from ..models.lightgbm_model import LightGBMForecaster
from ..models.losses import MultiTaskLoss
from ..models.tcn import TCNForecaster
from ..models.transformer import TransformerForecaster
from ..reproducibility import get_device, set_seed
from ..validation.diagnostics import export_fold_manifest
from ..validation.splits import PurgedWalkForwardCV
from .checkpointing import save_checkpoint
from .datasets import FinancialSequenceDataset
from .trainer import PyTorchTrainer

logger = logging.getLogger("alpha_volatility.training.experiment")


class ExperimentRunner:
    """Manages full execution of model training across walk-forward folds."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.output_dir = Path(config.artifacts.get("output_dir", "artifacts"))
        self.checkpoints_dir = Path(config.artifacts.get("checkpoints_dir", "checkpoints"))
        self.manifests_dir = Path(config.artifacts.get("manifests_dir", "artifacts/manifests"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        set_seed(config.seed)

    def run_model_training(
        self,
        model_name: str,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Runs walk-forward training for specified model ('tcn', 'transformer', 'lightgbm').

        Returns:
          (val_predictions_df, test_predictions_df, experiment_manifest)
        """
        logger.info("Initializing walk-forward experiment for model: %s", model_name)
        start_time = time.time()

        val_cfg = self.config.validation.get("protocol", {})
        self.config.validation.get("purging", {})
        embargo_cfg = self.config.validation.get("embargo", {})

        pwf = PurgedWalkForwardCV(
            n_splits=val_cfg.get("num_walk_forward_folds", 4),
            holdout_ratio=val_cfg.get("holdout_test_ratio", 0.20),
            embargo_bars=embargo_cfg.get("embargo_bars", 12),
            min_train_ratio=val_cfg.get("min_train_ratio", 0.40),
            window_type=val_cfg.get("train_window_type", "expanding"),
        )

        dev_df, test_df = pwf.split_holdout(df)
        folds = pwf.generate_folds(dev_df)

        # Export fold manifest
        export_fold_manifest(
            folds,
            output_path=self.manifests_dir / "fold_manifest.json",
            seed=self.config.seed,
        )

        feat_cols = FeaturePipeline().feature_columns
        seq_len = self.config.features.get("sequence", {}).get("sequence_length", 24)

        fold_val_preds: List[pd.DataFrame] = []

        # Walk-forward cross validation loop
        for fold in folds:
            logger.info("--- Processing %s (Train: %d, Val: %d) ---", fold.fold_id, len(fold.train_df), len(fold.val_df))

            # 1. Fit scaler exclusively on training fold
            scaler = FeaturePipeline.fit_scaler(fold.train_df, feat_cols, method="standard")
            scaled_train = FeaturePipeline.transform_features(fold.train_df, scaler, feat_cols)
            scaled_val = FeaturePipeline.transform_features(fold.val_df, scaler, feat_cols)

            if model_name == "lightgbm":
                lgb_params = self.config.models.get("lightgbm", {})
                forecaster = LightGBMForecaster(feature_names=feat_cols, params=lgb_params)
                forecaster.fit(scaled_train, scaled_val)
                predictor = UnifiedPredictor(forecaster, model_type="lightgbm", feature_names=feat_cols)
                val_pred = predictor.predict_dataframe(scaled_val)
                fold_val_preds.append(val_pred)

            elif model_name in ["tcn", "transformer"]:
                train_ds = FinancialSequenceDataset(scaled_train, feat_cols, seq_len=seq_len)
                val_ds = FinancialSequenceDataset(scaled_val, feat_cols, seq_len=seq_len)

                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

                if model_name == "tcn":
                    tcn_cfg = self.config.models.get("tcn", {})
                    model = TCNForecaster(
                        input_dim=len(feat_cols),
                        channel_sizes=tcn_cfg.get("channel_sizes", [64, 64, 64]),
                        kernel_size=tcn_cfg.get("kernel_size", 3),
                        dropout=tcn_cfg.get("dropout", 0.2),
                    )
                else:
                    tf_cfg = self.config.models.get("transformer", {})
                    model = TransformerForecaster(
                        input_dim=len(feat_cols),
                        d_model=tf_cfg.get("d_model", 64),
                        nhead=tf_cfg.get("nhead", 4),
                        num_layers=tf_cfg.get("num_layers", 3),
                        dim_feedforward=tf_cfg.get("dim_feedforward", 128),
                        dropout=tf_cfg.get("dropout", 0.15),
                    )

                train_cfg = self.config.models.get("training", {})
                loss_weights = train_cfg.get("loss_weights", {})
                loss_fn = MultiTaskLoss(
                    return_distribution="student_t",
                    weight_return=loss_weights.get("return_task", 1.0),
                    weight_volatility=loss_weights.get("volatility_task", 0.5),
                )

                trainer = PyTorchTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    learning_rate=train_cfg.get("learning_rate", 0.001),
                    max_epochs=train_cfg.get("max_epochs", 15),
                    patience=train_cfg.get("patience", 4),
                )
                trainer.fit(train_loader, val_loader)

                # Save fold checkpoint
                save_checkpoint(
                    {
                        "model_state": model.state_dict(),
                        "model_name": model_name,
                        "fold_id": fold.fold_id,
                        "feature_cols": feat_cols,
                        "scaler": scaler,
                    },
                    self.checkpoints_dir / f"{model_name}_{fold.fold_id}.pt",
                )

                predictor = UnifiedPredictor(
                    model,
                    model_type=model_name,
                    feature_names=feat_cols,
                    sequence_length=seq_len,
                )
                val_pred = predictor.predict_dataframe(scaled_val)
                fold_val_preds.append(val_pred)

        all_val_preds = pd.concat(fold_val_preds, ignore_index=True)
        all_val_preds = all_val_preds.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)

        # Final Holdout Test Evaluation:
        # Fit scaler on full dev set, train model on full dev set, evaluate on untouched test set
        logger.info("Training final model on full development set for holdout test evaluation...")
        scaler_full = FeaturePipeline.fit_scaler(dev_df, feat_cols, method="standard")
        scaled_dev = FeaturePipeline.transform_features(dev_df, scaler_full, feat_cols)
        scaled_test = FeaturePipeline.transform_features(test_df, scaler_full, feat_cols)

        if model_name == "lightgbm":
            lgb_params = self.config.models.get("lightgbm", {})
            final_model = LightGBMForecaster(feature_names=feat_cols, params=lgb_params)
            final_model.fit(scaled_dev)
            final_pred = UnifiedPredictor(final_model, model_type="lightgbm", feature_names=feat_cols)
            test_preds = final_pred.predict_dataframe(scaled_test)
        else:
            dev_ds = FinancialSequenceDataset(scaled_dev, feat_cols, seq_len=seq_len)
            dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=True)

            if model_name == "tcn":
                tcn_cfg = self.config.models.get("tcn", {})
                final_net = TCNForecaster(
                    input_dim=len(feat_cols),
                    channel_sizes=tcn_cfg.get("channel_sizes", [64, 64, 64]),
                    kernel_size=tcn_cfg.get("kernel_size", 3),
                    dropout=tcn_cfg.get("dropout", 0.2),
                )
            else:
                tf_cfg = self.config.models.get("transformer", {})
                final_net = TransformerForecaster(
                    input_dim=len(feat_cols),
                    d_model=tf_cfg.get("d_model", 64),
                    nhead=tf_cfg.get("nhead", 4),
                    num_layers=tf_cfg.get("num_layers", 3),
                    dim_feedforward=tf_cfg.get("dim_feedforward", 128),
                    dropout=tf_cfg.get("dropout", 0.15),
                )

            train_cfg = self.config.models.get("training", {})
            loss_weights = train_cfg.get("loss_weights", {})
            loss_fn = MultiTaskLoss(
                return_distribution="student_t",
                weight_return=loss_weights.get("return_task", 1.0),
                weight_volatility=loss_weights.get("volatility_task", 0.5),
            )
            trainer = PyTorchTrainer(
                model=final_net,
                loss_fn=loss_fn,
                learning_rate=train_cfg.get("learning_rate", 0.001),
                max_epochs=train_cfg.get("max_epochs", 15),
            )
            trainer.fit(dev_loader)

            save_checkpoint(
                {
                    "model_state": final_net.state_dict(),
                    "model_name": model_name,
                    "is_final": True,
                    "scaler": scaler_full,
                },
                self.checkpoints_dir / f"{model_name}_final.pt",
            )

            final_pred = UnifiedPredictor(
                final_net,
                model_type=model_name,
                feature_names=feat_cols,
                sequence_length=seq_len,
            )
            test_preds = final_pred.predict_dataframe(scaled_test)

        elapsed = time.time() - start_time
        manifest = {
            "experiment_id": self.config.experiment_id,
            "model_name": model_name,
            "seed": self.config.seed,
            "elapsed_seconds": round(elapsed, 2),
            "num_val_predictions": len(all_val_preds),
            "num_test_predictions": len(test_preds),
            "device": str(get_device()),
            "torch_version": torch.__version__,
        }

        # Save manifest
        with open(self.manifests_dir / f"experiment_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return all_val_preds, test_preds, manifest
