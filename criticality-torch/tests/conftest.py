"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42
