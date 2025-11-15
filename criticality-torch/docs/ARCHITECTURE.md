# Architecture Overview

This document describes the architecture of criticality-torch.

## Package Structure

```
criticality_torch/
├── data/                    # Data handling and preprocessing
│   ├── bayesian_network.py  # Bayesian network implementation
│   ├── dataset.py           # PyTorch Dataset class
│   └── preprocessing.py     # Data preprocessing utilities
├── models/                  # Neural network models
│   ├── base.py             # Base model architecture
│   ├── ensemble.py         # Ensemble implementation
│   └── losses.py           # Custom loss functions
├── training/               # Training infrastructure
│   ├── trainer.py          # Training loop
│   ├── callbacks.py        # Training callbacks
│   └── optimizer_weights.py # Ensemble weight optimization
├── evaluation/             # Evaluation and prediction
│   ├── predict.py          # Prediction pipeline
│   ├── risk.py            # Risk estimation
│   └── metrics.py         # Evaluation metrics
└── utils/                  # Utilities
    ├── config.py           # Configuration management
    ├── logging.py          # Logging setup
    └── visualization.py    # Plotting utilities
```

## Key Components

### 1. Bayesian Network (`data/bayesian_network.py`)

Models facility operations and criticality safety parameters using:
- Categorical distributions for discrete parameters (operations, controls, forms)
- Fitted continuous distributions (gamma, normal, log-normal, Weibull, GEV)
- Conditional probability tables
- Efficient sampling with multiprocessing

**Key methods:**
- `from_csv()`: Build network from facility data
- `fit()`: Fit probability distributions
- `sample()`: Generate Monte Carlo samples

### 2. Neural Network Ensemble (`models/ensemble.py`)

Ensemble of deep neural networks for keff prediction:
- Multiple independently trained models
- Optimized weighted combination
- Checkpoint management for best epoch selection
- Weight optimization using Nelder-Mead, BFGS, and Simulated Annealing

**Key methods:**
- `train()`: Train ensemble on dataset
- `predict()`: Generate weighted ensemble predictions
- `save()`/`load()`: Model persistence

### 3. Data Preprocessing (`data/preprocessing.py`)

Handles data transformation:
- One-hot encoding of categorical variables
- Centering and scaling of continuous variables
- Train/test splitting with mass-based stratification
- Feature engineering (volume, concentration calculation)

**Key methods:**
- `load_and_prepare()`: Load CSV and create train/test sets
- `transform()`: Transform new data using fitted parameters
- `save()`/`load()`: Preprocessor persistence

### 4. Training Infrastructure (`training/`)

PyTorch training loop with:
- Batch processing with DataLoader
- Validation split
- Metric tracking (loss, MAE)
- Callback system for checkpointing and visualization
- GPU support

**Key classes:**
- `Trainer`: Main training loop
- `CheckpointCallback`: Save model checkpoints
- `PlotCallback`: Generate training plots

### 5. Risk Estimation (`evaluation/risk.py`)

Monte Carlo-based risk estimation:
- Sample from Bayesian network
- Predict keff values using ensemble
- Calculate exceedance probability (keff >= USL)
- Aggregate over multiple iterations for confidence intervals

**Key methods:**
- `estimate()`: Perform full risk estimation
- Automatic result caching and loading

## Data Flow

1. **Training Flow:**
   ```
   CSV Data → Preprocessing → Dataset → DataLoader →
   Training Loop → Model Checkpoints → Weight Optimization →
   Ensemble Model
   ```

2. **Risk Estimation Flow:**
   ```
   Facility Data → Bayesian Network →
   Monte Carlo Samples → Preprocessing →
   Ensemble Prediction → Risk Calculation
   ```

## Key Design Decisions

### 1. PyTorch Native
- All models use `torch.nn.Module`
- Training uses PyTorch DataLoader and optimizers
- Full control over training loop

### 2. Modular Architecture
- Each component is independently testable
- Clear separation of concerns
- Easy to extend or replace components

### 3. Configuration Management
- YAML-based configuration
- Default configurations provided
- Easy to version control experiments

### 4. Checkpoint Strategy
- Save models at each epoch during retraining
- Select best epoch based on combined train/val MAE
- Enables post-hoc analysis and model selection

### 5. Weight Optimization
- Try multiple algorithms (Nelder-Mead, BFGS, SA)
- Incremental ensemble size testing
- Automatic selection of best approach

## Performance Considerations

### Memory Management
- Batch processing for large datasets
- Automatic sample size adjustment for memory constraints
- Efficient numpy/torch tensor conversions

### Parallelization
- Multiprocessing for BN sampling
- GPU acceleration for neural network training/inference
- Batch prediction for efficiency

### Caching
- Save preprocessor state to avoid refitting
- Cache trained models and checkpoints
- Optional result caching for risk estimation

## Extension Points

### Adding New Distributions
Extend `BayesianNetwork._fit_continuous()` with additional scipy.stats distributions.

### Custom Loss Functions
Add to `models/losses.py` following the `nn.Module` pattern.

### New Optimizers
Extend `models/base.get_optimizer()` with additional PyTorch optimizers.

### Additional Callbacks
Create new callback classes in `training/callbacks.py`.

## Testing Strategy

- Unit tests for individual components
- Integration tests for full workflows
- Fixtures for reproducible test data
- Coverage requirements: >80%

## Future Enhancements

Potential areas for expansion:
- Distributed training (PyTorch DDP)
- Mixed precision training
- TensorBoard integration
- MLflow experiment tracking
- Automated hyperparameter tuning
- Additional neural architectures (ResNets, Transformers)
