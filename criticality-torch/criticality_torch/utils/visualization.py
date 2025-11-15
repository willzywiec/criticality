"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11


def plot_training_history(
    history: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot training history.

    Args:
        history: DataFrame with training history
        save_path: Optional path to save plot
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Plot MAE
    ax.plot(history["epoch"], history["val_mae"],
            label="cross-validation", color="gray", linewidth=2)
    ax.plot(history["epoch"], history["mae"],
            label="training", color="black", linewidth=2)

    # Mark minimum
    min_idx = history["mae"].idxmin()
    min_epoch = history.loc[min_idx, "epoch"]
    min_mae = history.loc[min_idx, "mae"]

    ax.scatter(min_epoch, min_mae, color="red",
              label="training minimum", s=50, zorder=5)
    ax.annotate(
        f"{min_mae:.3e}",
        (min_epoch, min_mae),
        textcoords="offset points",
        xytext=(0, -15),
        ha="center",
        color="red",
        fontsize=9
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs. True Values",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot predictions vs. true values.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Optional path to save plot
        show: Whether to display plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    ax1.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', linewidth=2, label="Perfect prediction")
    ax1.set_xlabel("True keff")
    ax1.set_ylabel("Predicted keff")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residual plot
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel("True keff")
    ax2.set_ylabel("Residual (Predicted - True)")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Prediction plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_risk_distribution(
    risk_samples: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot risk distribution.

    Args:
        risk_samples: Array of risk estimates
        save_path: Optional path to save plot
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    ax.hist(risk_samples, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(risk_samples), color='r',
               linestyle='--', linewidth=2, label=f"Mean: {np.mean(risk_samples):.3e}")
    ax.set_xlabel("Risk Estimate")
    ax.set_ylabel("Frequency")
    ax.set_title("Process Criticality Accident Risk Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Risk distribution plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
