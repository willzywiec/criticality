# criticality-torch

A PyTorch-based implementation for modeling fissile material operations in nuclear facilities.

This package provides tools for:
- Building Bayesian networks to model facility operations and criticality safety parameters
- Training deep neural network ensembles to predict keff (effective neutron multiplication factor)
- Estimating process criticality accident risk through Monte Carlo sampling

## Overview

`criticality-torch` is a modern PyTorch port of the R package [criticality](https://github.com/willzywiec/criticality), with improvements including:

- **Native PyTorch**: Full PyTorch implementation replacing Keras/TensorFlow
- **Improved architecture**: Bug fixes and enhanced neural network construction
- **Better performance**: Optimized data loading, training, and inference pipelines
- **Modern tooling**: Type hints, comprehensive logging, and configuration management
- **Extensible design**: Modular architecture for easy customization

## Installation

```bash
pip install criticality-torch
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from criticality_torch import BayesianNetwork, NeuralEnsemble, RiskEstimator

# Create Bayesian network from facility data
bn = BayesianNetwork.from_csv("facility.csv", distribution="gamma")

# Train neural network ensemble
ensemble = NeuralEnsemble(
    layers=[8192, 256, 256, 256, 256, 16],
    ensemble_size=5,
    epochs=1500,
    batch_size=8192
)
ensemble.train("mcnp-dataset.csv")

# Estimate criticality risk
risk = RiskEstimator(bn, ensemble)
results = risk.estimate(
    sample_size=1_000_000,
    risk_pool=100,
    usl=0.95
)

print(f"Risk: {results.mean_risk:.3e} ± {results.std_risk:.3e}")
```

## Features

### Bayesian Networks
- Probabilistic modeling of facility operations and controls
- Support for multiple distributions (gamma, normal, log-normal, Weibull, GEV)
- Conditional probability table construction
- Parallel sampling for efficient inference

### Neural Network Ensembles
- Deep feedforward networks with customizable architectures
- Ensemble training with automatic weight optimization
- Multiple optimization algorithms (Nelder-Mead, BFGS, Simulated Annealing)
- Checkpoint management and model persistence
- Training visualization and metrics tracking

### Risk Assessment
- Monte Carlo-based risk estimation
- Parallel prediction pipeline
- Configurable cutoff values and sampling parameters
- Comprehensive result reporting with confidence intervals

## Documentation

Full documentation is available at [link to docs].

## Citation

If you use this package in your research, please cite:

Zywiec et al. (2021). "Modeling fissile material operations in nuclear facilities."
*Reliability Engineering & System Safety*, 207, 107322.
doi:10.1016/j.ress.2020.107322

## License

MIT License - see LICENSE file for details.

## Related Projects

- [criticality (R package)](https://github.com/willzywiec/criticality) - Original R implementation
