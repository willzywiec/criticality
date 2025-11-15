"""Evaluation metrics for criticality models."""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate mean absolute error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()


def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate root mean squared error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return torch.sqrt(torch.mean(torch.pow(y_true - y_pred, 2))).item()


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate R-squared coefficient of determination.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² value
    """
    ss_res = torch.sum(torch.pow(y_true - y_pred, 2))
    ss_tot = torch.sum(torch.pow(y_true - torch.mean(y_true), 2))
    return (1 - ss_res / ss_tot).item()


def evaluate_model(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on test data.

    Args:
        model: Model to evaluate
        X: Input features
        y: True targets
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_device = X.to(device)
        y_pred = model(X_device).cpu()

    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": root_mean_squared_error(y, y_pred),
        "r2": r_squared(y, y_pred),
    }

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics
