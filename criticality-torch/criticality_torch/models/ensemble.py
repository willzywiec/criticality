"""Neural network ensemble for criticality prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from criticality_torch.models.base import CriticalityModel, get_optimizer
from criticality_torch.models.losses import get_loss_function
from criticality_torch.data.dataset import CriticalityDataset
from criticality_torch.data.preprocessing import DataPreprocessor
from criticality_torch.training.trainer import Trainer, TrainingHistory
from criticality_torch.training.callbacks import CheckpointCallback, PlotCallback
from criticality_torch.training.optimizer_weights import (
    EnsembleWeightOptimizer,
    compute_ensemble_metrics
)

logger = logging.getLogger(__name__)


class NeuralEnsemble:
    """Ensemble of neural networks for keff prediction.

    Args:
        layers: List of layer sizes or layer string (e.g., "8192-256-256-256-256-16")
        ensemble_size: Number of models in ensemble
        epochs: Training epochs
        batch_size: Training batch size
        loss: Loss function name
        optimizer_name: Optimizer name
        learning_rate: Learning rate
        val_split: Validation split fraction
        device: Device to train on
    """

    def __init__(
        self,
        layers: List[int] | str = "8192-256-256-256-256-16",
        ensemble_size: int = 5,
        epochs: int = 1500,
        batch_size: int = 8192,
        loss: str = "sse",
        optimizer_name: str = "adamax",
        learning_rate: float = 0.00075,
        val_split: float = 0.2,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize ensemble."""
        self.layer_spec = layers if isinstance(layers, str) else "-".join(map(str, layers))
        self.hidden_layers = (
            [int(x) for x in layers.split("-")] if isinstance(layers, str) else layers
        )
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_name = loss
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensemble components
        self.models: List[nn.Module] = []
        self.weights: Optional[np.ndarray] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.training_histories: List[TrainingHistory] = []

        # Metrics
        self.test_mae: Optional[float] = None
        self.ensemble_test_mae: Optional[float] = None

        logger.info(f"Initialized ensemble with {ensemble_size} models on {self.device}")

    def train(
        self,
        csv_path: str,
        code: str = "mcnp",
        output_dir: Optional[str] = None,
        retrain_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        """Train ensemble on data from CSV.

        Args:
            csv_path: Path to training data CSV
            code: Monte Carlo code name
            output_dir: Directory for outputs
            retrain_epochs: Epochs for retraining (default: epochs/10)
            verbose: Whether to show progress
        """
        if output_dir is None:
            output_dir = "./training"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data
        logger.info("Loading and preprocessing data...")
        self.preprocessor = DataPreprocessor(code=code)
        X_train, y_train, X_test, y_test = self.preprocessor.load_and_prepare(
            csv_path, output_dir=str(output_path)
        )

        # Create datasets
        full_train_dataset = CriticalityDataset(X_train, y_train)
        test_dataset = CriticalityDataset(X_test, y_test)

        # Split into train/val
        train_size = int((1 - self.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Get input dimension
        input_dim = full_train_dataset.num_features

        # Train ensemble members
        logger.info(f"Training {self.ensemble_size} ensemble members...")
        self._train_ensemble_members(
            input_dim, train_loader, val_loader, output_path, verbose
        )

        # Retrain with checkpoints
        if retrain_epochs is None:
            retrain_epochs = self.epochs // 10

        logger.info(f"Retraining with checkpointing ({retrain_epochs} epochs)...")
        self._retrain_with_checkpoints(
            input_dim, train_loader, val_loader, output_path, retrain_epochs, verbose
        )

        # Optimize weights
        logger.info("Optimizing ensemble weights...")
        self._optimize_weights(test_loader, output_path)

        # Save ensemble
        self.save(output_path / "ensemble.pt")

        logger.info("Training complete!")

    def _train_ensemble_members(
        self,
        input_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_path: Path,
        verbose: bool
    ) -> None:
        """Train individual ensemble members.

        Args:
            input_dim: Number of input features
            train_loader: Training data loader
            val_loader: Validation data loader
            output_path: Output directory
            verbose: Show progress
        """
        model_dir = output_path / "model"
        model_dir.mkdir(exist_ok=True)

        for i in range(1, self.ensemble_size + 1):
            logger.info(f"\nTraining model {i}/{self.ensemble_size}")

            # Check if model already exists
            model_path = model_dir / f"{i}.pt"
            if model_path.exists():
                logger.info(f"Model {i} already trained, loading...")
                model = CriticalityModel(input_dim, self.hidden_layers)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.models.append(model)
                continue

            # Create model
            model = CriticalityModel(input_dim, self.hidden_layers)
            optimizer = get_optimizer(model, self.optimizer_name, self.learning_rate)
            loss_fn = get_loss_function(self.loss_name)

            # Create trainer with callbacks
            callbacks = [
                PlotCallback(plot_dir=str(model_dir), model_id=i)
            ]
            trainer = Trainer(model, optimizer, loss_fn, self.device, callbacks)

            # Train
            history = trainer.fit(train_loader, val_loader, self.epochs, verbose=verbose)
            self.training_histories.append(history)

            # Save model
            torch.save(model.state_dict(), model_path)
            self.models.append(model)

    def _retrain_with_checkpoints(
        self,
        input_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_path: Path,
        retrain_epochs: int,
        verbose: bool
    ) -> None:
        """Retrain models with epoch checkpoints.

        Args:
            input_dim: Number of input features
            train_loader: Training data loader
            val_loader: Validation data loader
            output_path: Output directory
            retrain_epochs: Number of retraining epochs
            verbose: Show progress
        """
        remodel_dir = output_path / "remodel"
        remodel_dir.mkdir(exist_ok=True)

        for i in range(1, self.ensemble_size + 1):
            logger.info(f"\nRetraining model {i}/{self.ensemble_size}")

            # Check if checkpoints already exist
            checkpoints = list(remodel_dir.glob(f"{i}-*.pt"))
            if len(checkpoints) >= retrain_epochs:
                logger.info(f"Model {i} checkpoints already exist, skipping...")
                continue

            # Create model and load weights
            model = self.models[i - 1]
            optimizer = get_optimizer(model, self.optimizer_name, self.learning_rate)
            loss_fn = get_loss_function(self.loss_name)

            # Create trainer with checkpoint callback
            callbacks = [
                CheckpointCallback(
                    checkpoint_dir=str(remodel_dir),
                    model_id=i,
                    monitor="val_mae",
                    save_best_only=False
                ),
                PlotCallback(plot_dir=str(remodel_dir), model_id=i)
            ]
            trainer = Trainer(model, optimizer, loss_fn, self.device, callbacks)

            # Retrain
            history = trainer.fit(
                train_loader, val_loader, retrain_epochs, verbose=verbose
            )

    def _optimize_weights(
        self,
        test_loader: DataLoader,
        output_path: Path
    ) -> None:
        """Optimize ensemble weights.

        Args:
            test_loader: Test data loader
            output_path: Output directory
        """
        remodel_dir = output_path / "remodel"

        # Load best checkpoint for each model based on minimum MAE
        best_models = []
        best_epochs = []

        for i in range(1, self.ensemble_size + 1):
            # Read training history
            history_path = remodel_dir / f"{i}.csv"
            history = pd.read_csv(history_path)

            # Find epoch with minimum combined MAE
            history["combined_mae"] = history["mae"] + history["val_mae"]
            best_epoch = history.loc[history["combined_mae"].idxmin(), "epoch"]
            best_epochs.append(int(best_epoch))

            # Load checkpoint
            checkpoint_path = remodel_dir / f"{i}-{int(best_epoch)}.pt"
            model = CriticalityModel(
                self.models[0].input_dim,
                self.hidden_layers
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.to(self.device)
            best_models.append(model)

        # Collect test predictions
        test_predictions = []
        test_targets = None

        for X_batch, y_batch in test_loader:
            batch_preds = []
            for model in best_models:
                model.eval()
                with torch.no_grad():
                    X_device = X_batch.to(self.device)
                    pred = model(X_device).cpu().numpy()
                    batch_preds.append(pred)

            test_predictions.append(batch_preds)

            if test_targets is None:
                test_targets = y_batch.numpy()
            else:
                test_targets = np.concatenate([test_targets, y_batch.numpy()])

        # Concatenate predictions
        ensemble_predictions = []
        for i in range(self.ensemble_size):
            model_preds = np.concatenate([batch[i] for batch in test_predictions])
            ensemble_predictions.append(model_preds.flatten())

        test_targets = test_targets.flatten()

        # Optimize weights
        optimizer = EnsembleWeightOptimizer(ensemble_predictions, test_targets)
        weights_list, mae_list, methods_list, best_n = optimizer.optimize_incremental(
            max_models=self.ensemble_size
        )

        # Save test MAE results
        test_mae_df = pd.DataFrame({
            "avg": mae_list,
            "method": methods_list
        })
        test_mae_df.to_csv(output_path / "test-mae.csv", index=False)

        # Use best weights
        self.weights = weights_list[best_n - 1]
        self.models = best_models[:best_n]

        # Calculate final metrics
        individual_maes = [
            np.mean(np.abs(test_targets - pred))
            for pred in ensemble_predictions[:best_n]
        ]
        weighted_pred = sum(
            w * p for w, p in zip(self.weights, ensemble_predictions[:best_n])
        )
        ensemble_mae = np.mean(np.abs(test_targets - weighted_pred))

        self.test_mae = np.mean(individual_maes)
        self.ensemble_test_mae = ensemble_mae

        logger.info(f"Mean individual test MAE: {self.test_mae:.6f}")
        logger.info(f"Ensemble test MAE: {self.ensemble_test_mae:.6f}")
        logger.info(f"Best ensemble size: {best_n}")
        logger.info(f"Weights: {self.weights}")

    def predict(
        self,
        X: torch.Tensor,
        return_individual: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate ensemble predictions.

        Args:
            X: Input features
            return_individual: Whether to return individual model predictions

        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        if self.weights is None:
            raise ValueError("Ensemble not trained yet")

        individual_preds = []

        for model in self.models:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                X_device = X.to(self.device)
                pred = model(X_device).cpu()
                individual_preds.append(pred)

        # Weight and combine predictions
        ensemble_pred = sum(
            w * pred for w, pred in zip(self.weights, individual_preds)
        )

        if return_individual:
            return ensemble_pred, individual_preds
        else:
            return ensemble_pred

    def save(self, path: str) -> None:
        """Save ensemble to file.

        Args:
            path: Save path
        """
        state = {
            "layer_spec": self.layer_spec,
            "hidden_layers": self.hidden_layers,
            "ensemble_size": len(self.models),
            "model_states": [model.state_dict() for model in self.models],
            "weights": self.weights,
            "hyperparameters": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "loss": self.loss_name,
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "val_split": self.val_split,
            }
        }

        torch.save(state, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        input_dim: int,
        device: Optional[torch.device] = None
    ) -> "NeuralEnsemble":
        """Load ensemble from file.

        Args:
            path: Path to saved ensemble
            input_dim: Number of input features
            device: Device to load to

        Returns:
            Loaded ensemble
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)

        # Create ensemble
        ensemble = cls(
            layers=checkpoint["layer_spec"],
            ensemble_size=checkpoint["ensemble_size"],
            device=device,
            **checkpoint["hyperparameters"]
        )

        # Load models
        ensemble.models = []
        for state_dict in checkpoint["model_states"]:
            model = CriticalityModel(input_dim, ensemble.hidden_layers)
            model.load_state_dict(state_dict)
            model.to(device)
            ensemble.models.append(model)

        ensemble.weights = checkpoint["weights"]

        logger.info(f"Ensemble loaded from {path}")
        return ensemble
