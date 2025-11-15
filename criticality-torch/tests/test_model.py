"""Tests for model architecture."""

import pytest
import torch
from criticality_torch.models.base import CriticalityModel, get_optimizer


def test_model_creation():
    """Test basic model creation."""
    model = CriticalityModel(input_dim=10, hidden_layers=[64, 32])

    assert model.input_dim == 10
    assert model.hidden_layers == [64, 32]

    # Test forward pass
    x = torch.randn(5, 10)
    y = model(x)

    assert y.shape == (5, 1)


def test_model_from_layer_string():
    """Test model creation from layer string."""
    model = CriticalityModel.from_layer_string(
        input_dim=10,
        layer_string="64-32-16"
    )

    assert model.hidden_layers == [64, 32, 16]

    # Test forward pass
    x = torch.randn(3, 10)
    y = model(x)

    assert y.shape == (3, 1)


def test_deep_model():
    """Test deep model with many layers (bug fix verification)."""
    # This tests the fix for the R code bug
    layers = [256, 256, 256, 256, 256, 16]
    model = CriticalityModel(input_dim=50, hidden_layers=layers)

    # Count layers in network
    linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]

    # Should have len(layers) + 1 (including output layer)
    assert len(linear_layers) == len(layers) + 1

    # Verify layer sizes
    assert linear_layers[0].in_features == 50
    assert linear_layers[0].out_features == 256
    assert linear_layers[-1].out_features == 1


def test_optimizer_creation():
    """Test optimizer creation."""
    model = CriticalityModel(input_dim=10, hidden_layers=[32])

    # Test different optimizers
    optimizers = ["adam", "adamax", "sgd", "rmsprop"]

    for opt_name in optimizers:
        optimizer = get_optimizer(model, opt_name, learning_rate=0.001)
        assert optimizer is not None
        assert len(optimizer.param_groups) > 0


def test_invalid_activation():
    """Test that invalid activation raises error."""
    with pytest.raises(ValueError):
        CriticalityModel(input_dim=10, hidden_layers=[32], activation="invalid")


def test_invalid_optimizer():
    """Test that invalid optimizer raises error."""
    model = CriticalityModel(input_dim=10, hidden_layers=[32])

    with pytest.raises(ValueError):
        get_optimizer(model, "invalid_optimizer", learning_rate=0.001)
