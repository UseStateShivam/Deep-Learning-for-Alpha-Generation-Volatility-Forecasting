"""
Neural Model Trainer
Orchestrates PyTorch optimization, early stopping, and metric logging.
"""

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.losses import MultiTaskLoss
from ..reproducibility import get_device
from .callbacks import EarlyStopping

logger = logging.getLogger("alpha_volatility.training.trainer")


class PyTorchTrainer:
    """Standardized trainer for TCN and Transformer architectures."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultiTaskLoss,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        patience: int = 5,
        grad_clip_norm: float = 1.0,
        device: str = "auto",
    ):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.patience = patience

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=learning_rate * 0.05,
        )
        self.early_stopping = EarlyStopping(patience=patience, mode="min")
        self.history: List[Dict[str, float]] = []
        self.best_model_weights: Optional[Dict] = None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_ret_loss = 0.0
        total_vol_loss = 0.0
        n_batches = 0

        for x_batch, y_ret_batch, y_vol_batch in dataloader:
            x_b = x_batch.to(self.device)
            y_r = y_ret_batch.to(self.device)
            y_v = y_vol_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_b)
            loss, metrics = self.loss_fn(y_r, y_v, preds)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            total_loss += metrics["loss_total"]
            total_ret_loss += metrics["loss_return"]
            total_vol_loss += metrics["loss_volatility"]
            n_batches += 1

        denom = max(1, n_batches)
        return {
            "train_loss": total_loss / denom,
            "train_ret_loss": total_ret_loss / denom,
            "train_vol_loss": total_vol_loss / denom,
        }

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs evaluation over validation DataLoader."""
        self.model.eval()
        total_loss = 0.0
        total_ret_loss = 0.0
        total_vol_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_ret_batch, y_vol_batch in dataloader:
                x_b = x_batch.to(self.device)
                y_r = y_ret_batch.to(self.device)
                y_v = y_vol_batch.to(self.device)

                preds = self.model(x_b)
                loss, metrics = self.loss_fn(y_r, y_v, preds)

                total_loss += metrics["loss_total"]
                total_ret_loss += metrics["loss_return"]
                total_vol_loss += metrics["loss_volatility"]
                n_batches += 1

        denom = max(1, n_batches)
        return {
            "val_loss": total_loss / denom,
            "val_ret_loss": total_ret_loss / denom,
            "val_vol_loss": total_vol_loss / denom,
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> List[Dict[str, float]]:
        """Executes full training loop with early stopping."""
        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                **train_metrics,
            }

            if val_loader is not None and len(val_loader) > 0:
                val_metrics = self.validate_epoch(val_loader)
                epoch_record.update(val_metrics)
                val_score = val_metrics["val_loss"]

                is_best = self.early_stopping(val_score)
                if is_best:
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())

                if self.early_stopping.early_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break
            else:
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

            self.scheduler.step()
            self.history.append(epoch_record)

        # Restore best weights if available
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        return self.history
