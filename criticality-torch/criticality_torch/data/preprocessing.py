"""Data preprocessing utilities for criticality modeling."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle


class DataPreprocessor:
    """Preprocessor for criticality safety data.

    Handles:
    - Data loading from CSV
    - One-hot encoding of categorical variables
    - Centering and scaling of continuous variables
    - Train/test split
    - Data persistence

    Matches functionality from scale.R and tabulate.R
    """

    def __init__(
        self,
        code: str = "mcnp",
        test_split: float = 0.2,
        mass_threshold: float = 200.0,
    ) -> None:
        """Initialize preprocessor.

        Args:
            code: Monte Carlo code name (e.g., 'mcnp', 'cog')
            test_split: Fraction of data to use for testing
            mass_threshold: Mass threshold for test set selection
        """
        self.code = code.lower()
        self.test_split = test_split
        self.mass_threshold = mass_threshold

        # Scaling parameters (learned from training data)
        self.training_mean: Optional[Dict[str, float]] = None
        self.training_std: Optional[Dict[str, float]] = None

        # One-hot encoding mappings
        self.categorical_columns: List[str] = []
        self.continuous_columns: List[str] = ["mass", "rad", "thk", "vol", "conc"]

        # Dataset storage
        self.training_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def load_and_prepare(
        self,
        csv_path: str,
        output_dir: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load CSV data and prepare training/test sets.

        Args:
            csv_path: Path to CSV file with simulation data
            output_dir: Directory to save processed dataset

        Returns:
            Tuple of (X_train, y_train, X_test, y_test) as tensors
        """
        # Load data
        df = pd.read_csv(csv_path, encoding="utf-8")
        df = df.dropna()

        # Shuffle data
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Calculate derived features
        df = self._calculate_derived_features(df)

        # Identify categorical columns (excluding keff and sd which are targets)
        self.categorical_columns = [
            col for col in df.columns
            if df[col].dtype == "object" and col not in ["keff", "sd"]
        ]

        # One-hot encode categorical variables
        df_encoded = self._one_hot_encode(df)

        # Remove columns with single unique value
        df_encoded = self._remove_constant_columns(df_encoded)

        # Split into train/test
        self._split_data(df_encoded)

        # Scale continuous features
        X_train, y_train = self._scale_and_extract(self.training_data, fit=True)
        X_test, y_test = self._scale_and_extract(self.test_data, fit=False)

        # Save dataset if output directory provided
        if output_dir:
            self.save(output_dir)

        return X_train, y_train, X_test, y_test

    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume and concentration features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with added features
        """
        # Calculate volume (cc) - assuming spherical geometry
        df["vol"] = (4.0 / 3.0) * np.pi * np.power(df["rad"], 3)

        # Calculate concentration (g/cc)
        df["conc"] = df["mass"] / df["vol"]
        df["conc"] = df["conc"].replace([np.inf, -np.inf], 0)

        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with one-hot encoded categorical variables
        """
        # Separate target variables
        targets = df[["keff", "sd"]] if "sd" in df.columns else df[["keff"]]

        # Get features
        features = df.drop(columns=["keff", "sd"] if "sd" in df.columns else ["keff"])

        # One-hot encode
        encoded_features = pd.get_dummies(
            features,
            columns=self.categorical_columns,
            prefix_sep="",
            dtype=float
        )

        # Combine with targets
        result = pd.concat([encoded_features, targets], axis=1)

        return result

    def _remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with only one unique value.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with constant columns removed
        """
        # Don't remove target columns
        targets = ["keff", "sd"] if "sd" in df.columns else ["keff"]
        feature_cols = [col for col in df.columns if col not in targets]

        # Find columns with single unique value
        constant_cols = [
            col for col in feature_cols
            if df[col].nunique() <= 1
        ]

        if constant_cols:
            df = df.drop(columns=constant_cols)

        return df

    def _split_data(self, df: pd.DataFrame) -> None:
        """Split data into training and test sets.

        Uses mass threshold strategy from R implementation.

        Args:
            df: Input dataframe
        """
        # Filter test data based on mass threshold
        if "mass" in df.columns:
            test_mask = df["mass"] > self.mass_threshold
            test_candidates = df[test_mask]

            # Sample test set
            n_test = int(len(df) * self.test_split)
            if len(test_candidates) >= n_test:
                test_indices = test_candidates.sample(n=n_test, random_state=42).index
            else:
                # Fallback to random split if not enough high-mass samples
                test_indices = df.sample(n=n_test, random_state=42).index

            self.test_data = df.loc[test_indices].copy()
            self.training_data = df.drop(test_indices).copy()
        else:
            # Simple random split if no mass column
            n_test = int(len(df) * self.test_split)
            self.test_data = df.sample(n=n_test, random_state=42)
            self.training_data = df.drop(self.test_data.index)

    def _scale_and_extract(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale features and extract X, y tensors.

        Args:
            df: Input dataframe
            fit: Whether to fit scaling parameters

        Returns:
            Tuple of (X, y) tensors
        """
        df = df.copy()

        # Extract target
        y = df["keff"].values

        # Get features (excluding keff and sd)
        X_df = df.drop(columns=["keff", "sd"] if "sd" in df.columns else ["keff"])

        # Scale continuous features
        for col in self.continuous_columns:
            if col in X_df.columns:
                if fit:
                    if self.training_mean is None:
                        self.training_mean = {}
                        self.training_std = {}

                    self.training_mean[col] = X_df[col].mean()
                    self.training_std[col] = X_df[col].std()

                # Apply scaling
                if self.training_mean and self.training_std:
                    X_df[col] = (
                        (X_df[col] - self.training_mean[col]) / self.training_std[col]
                    )

        # Convert to tensors
        X = torch.tensor(X_df.values, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        return X, y

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform new data using fitted preprocessing.

        Args:
            df: Input dataframe

        Returns:
            Transformed feature tensor
        """
        if self.training_mean is None:
            raise ValueError("Preprocessor must be fitted before transforming data")

        df = df.copy()

        # Calculate derived features
        df = self._calculate_derived_features(df)

        # One-hot encode
        df_encoded = pd.get_dummies(
            df,
            columns=self.categorical_columns,
            prefix_sep="",
            dtype=float
        )

        # Ensure same columns as training data
        if self.training_data is not None:
            train_cols = [
                col for col in self.training_data.columns
                if col not in ["keff", "sd"]
            ]
            # Add missing columns
            for col in train_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            # Reorder columns
            df_encoded = df_encoded[train_cols]

        # Scale continuous features
        for col in self.continuous_columns:
            if col in df_encoded.columns:
                df_encoded[col] = (
                    (df_encoded[col] - self.training_mean[col]) /
                    self.training_std[col]
                )

        return torch.tensor(df_encoded.values, dtype=torch.float32)

    def save(self, output_dir: str) -> None:
        """Save preprocessor and processed data.

        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save preprocessor state
        state = {
            "code": self.code,
            "test_split": self.test_split,
            "mass_threshold": self.mass_threshold,
            "training_mean": self.training_mean,
            "training_std": self.training_std,
            "categorical_columns": self.categorical_columns,
            "continuous_columns": self.continuous_columns,
        }

        with open(output_path / f"{self.code}-preprocessor.pkl", "wb") as f:
            pickle.dump(state, f)

        # Save processed data
        if self.training_data is not None:
            self.training_data.to_csv(
                output_path / f"{self.code}-training-data.csv",
                index=False
            )
        if self.test_data is not None:
            self.test_data.to_csv(
                output_path / f"{self.code}-test-data.csv",
                index=False
            )

    @classmethod
    def load(cls, preprocessor_path: str) -> "DataPreprocessor":
        """Load preprocessor from file.

        Args:
            preprocessor_path: Path to saved preprocessor

        Returns:
            Loaded preprocessor instance
        """
        with open(preprocessor_path, "rb") as f:
            state = pickle.load(f)

        preprocessor = cls(
            code=state["code"],
            test_split=state["test_split"],
            mass_threshold=state["mass_threshold"]
        )

        preprocessor.training_mean = state["training_mean"]
        preprocessor.training_std = state["training_std"]
        preprocessor.categorical_columns = state["categorical_columns"]
        preprocessor.continuous_columns = state["continuous_columns"]

        return preprocessor
