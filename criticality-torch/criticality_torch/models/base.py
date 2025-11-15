"""Base neural network model for criticality prediction."""

import torch
import torch.nn as nn
from typing import List, Optional


class CriticalityModel(nn.Module):
    """Deep neural network for predicting keff values.

    This model creates a feedforward neural network with configurable
    hidden layer sizes, matching the architecture from model.R but
    with the bug fix for proper layer construction.

    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes (e.g., [8192, 256, 256, 256, 256, 16])
        activation: Activation function name ('relu', 'tanh', etc.)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "relu"
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(self._get_activation(activation))

        # Hidden layers - properly constructed (bug fix from R code)
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self._get_activation(activation))

        # Output layer (linear activation for regression)
        layers.append(nn.Linear(hidden_layers[-1], 1))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation function name

        Returns:
            Activation module
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }

        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")

        return activations[name.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x)

    @classmethod
    def from_layer_string(
        cls,
        input_dim: int,
        layer_string: str,
        activation: str = "relu"
    ) -> "CriticalityModel":
        """Create model from layer string specification.

        Matches the R interface where layers are specified as a string
        like "8192-256-256-256-256-16".

        Args:
            input_dim: Number of input features
            layer_string: Hyphen-separated layer sizes (e.g., "256-256-16")
            activation: Activation function name

        Returns:
            CriticalityModel instance
        """
        hidden_layers = [int(x) for x in layer_string.split("-")]
        return cls(input_dim, hidden_layers, activation)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float
) -> torch.optim.Optimizer:
    """Get optimizer by name.

    Maps R optimizer names to PyTorch optimizers.

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer
        learning_rate: Learning rate

    Returns:
        Optimizer instance
    """
    optimizer_map = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamax": torch.optim.Adamax,
        "nadam": torch.optim.NAdam,
        "rmsprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
    }

    opt_name = optimizer_name.lower()
    if opt_name not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer_class = optimizer_map[opt_name]

    # NAdam was added in PyTorch 1.11, use Adam as fallback
    if opt_name == "nadam":
        try:
            return optimizer_class(model.parameters(), lr=learning_rate)
        except AttributeError:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer_class(model.parameters(), lr=learning_rate)
