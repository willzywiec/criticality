"""PyTorch Dataset for criticality modeling."""

import torch
from torch.utils.data import Dataset
from typing import Tuple


class CriticalityDataset(Dataset):
    """PyTorch Dataset for criticality safety data.

    Args:
        X: Feature tensor of shape (n_samples, n_features)
        y: Target tensor of shape (n_samples, 1)
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Initialize dataset.

        Args:
            X: Feature tensor
            y: Target tensor
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length, got {len(X)} and {len(y)}"
            )

        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target)
        """
        return self.X[idx], self.y[idx]

    @property
    def num_features(self) -> int:
        """Get number of features.

        Returns:
            Number of input features
        """
        return self.X.shape[1]
