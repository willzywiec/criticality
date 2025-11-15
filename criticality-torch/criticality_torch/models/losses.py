"""Custom loss functions for criticality modeling."""

import torch
import torch.nn as nn
from typing import Optional


class SSELoss(nn.Module):
    """Sum of Squared Errors loss function.

    Equivalent to the SSE loss used in the R implementation:
    SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))
    """

    def __init__(self, reduction: str = "sum") -> None:
        """Initialize SSE loss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'sum' matches the R implementation.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute SSE loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            SSE loss value
        """
        squared_errors = torch.pow(y_true - y_pred, 2)

        if self.reduction == "sum":
            return torch.sum(squared_errors)
        elif self.reduction == "mean":
            return torch.mean(squared_errors)
        elif self.reduction == "none":
            return squared_errors
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def get_loss_function(loss_name: str) -> nn.Module:
    """Get loss function by name.

    Args:
        loss_name: Name of loss function ('sse', 'mse', 'mae')

    Returns:
        Loss function module
    """
    if loss_name.lower() == "sse":
        return SSELoss()
    elif loss_name.lower() == "mse":
        return nn.MSELoss()
    elif loss_name.lower() == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
