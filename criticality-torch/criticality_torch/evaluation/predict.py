"""Prediction module for generating keff predictions from Bayesian network samples."""

import torch
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from criticality_torch.data.bayesian_network import BayesianNetwork
    from criticality_torch.models.ensemble import NeuralEnsemble
    from criticality_torch.data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class Predictor:
    """Predictor for generating keff values from Bayesian network samples.

    Args:
        bayesian_network: Fitted Bayesian network
        ensemble: Trained neural ensemble
        preprocessor: Data preprocessor
    """

    def __init__(
        self,
        bayesian_network: "BayesianNetwork",
        ensemble: "NeuralEnsemble",
        preprocessor: "DataPreprocessor"
    ) -> None:
        """Initialize predictor.

        Args:
            bayesian_network: BN for generating samples
            ensemble: Trained ensemble for predictions
            preprocessor: Preprocessor for feature transformation
        """
        self.bn = bayesian_network
        self.ensemble = ensemble
        self.preprocessor = preprocessor

    def predict_from_samples(
        self,
        sample_size: int = 1000000,
        keff_cutoff: float = 0.9,
        mass_cutoff: float = 100.0,
        rad_cutoff: float = 7.0,
        n_cores: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate keff predictions from BN samples.

        Args:
            sample_size: Number of BN samples
            keff_cutoff: Initial keff threshold for filtering
            mass_cutoff: Mass threshold (g)
            rad_cutoff: Radius threshold (cm)
            n_cores: Number of CPU cores for sampling

        Returns:
            DataFrame with samples and predicted keff values
        """
        logger.info(f"Generating {sample_size} samples from Bayesian network...")

        # Sample from BN
        samples = self.bn.sample(
            n_samples=sample_size,
            mass_cutoff=mass_cutoff,
            rad_cutoff=rad_cutoff,
            n_cores=n_cores
        )

        logger.info(f"Processing {len(samples)} samples")

        # Calculate physical properties
        samples = self._calculate_physical_properties(samples)

        # Preprocess for neural network
        X = self.preprocessor.transform(samples)

        # Initial prediction for filtering
        if keff_cutoff > 0 and len(samples) > 1:
            logger.info("Performing initial keff filtering...")
            initial_pred = self._predict_single_model(X)
            keep_mask = initial_pred.flatten() >= keff_cutoff
            samples = samples[keep_mask].reset_index(drop=True)
            X = X[keep_mask]
            logger.info(f"Samples after filtering: {len(samples)}")

        # Full ensemble prediction
        if len(samples) > 0:
            logger.info("Generating ensemble predictions...")
            keff_pred = self.ensemble.predict(X)
            samples["keff"] = keff_pred.numpy().flatten()
        else:
            samples["keff"] = []

        return samples

    def _calculate_physical_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate physical properties from BN samples.

        Args:
            df: Sampled data

        Returns:
            DataFrame with calculated properties
        """
        df = df.copy()

        # Fissile material densities (g/cc)
        density_map = {
            "alpha": 19.86,
            "delta": 15.92,
            "puo2": 11.5,
            "heu": 18.85,
            "uo2": 10.97
        }

        densities = df["form"].map(density_map).fillna(19.86)

        # Calculate volume (cc) - spherical geometry
        vol = (4.0 / 3.0) * np.pi * np.power(df["rad"], 3)

        # Fix moderator if volume is too small
        min_vol = df["mass"] / densities
        needs_fix = vol <= min_vol
        df.loc[needs_fix, "mod"] = "none"
        vol[needs_fix] = min_vol[needs_fix]

        # Recalculate radius
        df["rad"] = np.power((3.0 / 4.0) * vol / np.pi, 1.0 / 3.0)

        # Fix reflector and thickness
        df.loc[df["thk"] == 0, "ref"] = "none"
        df.loc[df["ref"] == "none", "thk"] = 0

        # Calculate concentration
        df["conc"] = df["mass"] / vol
        df["conc"] = df["conc"].replace([np.inf, -np.inf], 0)

        # Store volume
        df["vol"] = vol

        return df

    def _predict_single_model(self, X: torch.Tensor) -> torch.Tensor:
        """Quick prediction using first ensemble model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        model = self.ensemble.models[0]
        model.eval()
        with torch.no_grad():
            X_device = X.to(self.ensemble.device)
            pred = model(X_device).cpu()
        return pred

    def predict_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10000
    ) -> np.ndarray:
        """Generate predictions for a dataframe.

        Args:
            df: Input dataframe with features
            batch_size: Batch size for prediction

        Returns:
            Array of predictions
        """
        # Calculate physical properties if needed
        if "vol" not in df.columns or "conc" not in df.columns:
            df = self._calculate_physical_properties(df)

        # Transform features
        X = self.preprocessor.transform(df)

        # Generate predictions in batches
        predictions = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            pred_batch = self.ensemble.predict(X_batch)
            predictions.append(pred_batch.numpy())

        return np.concatenate(predictions, axis=0).flatten()
