# Examples

This directory contains example scripts demonstrating the usage of criticality-torch.

## Prerequisites

Ensure you have installed the package:

```bash
pip install -e ..
```

## Examples

### 1. Build Bayesian Network

Build a Bayesian network from facility operations data:

```bash
python build_bayesian_network.py
```

This will:
- Load facility CSV data
- Fit probability distributions for all parameters
- Save the fitted Bayesian network
- Generate and display sample statistics

### 2. Train Neural Ensemble

Train an ensemble of neural networks to predict keff values:

```bash
python train_ensemble.py
```

This will:
- Load and preprocess Monte Carlo simulation data
- Train multiple neural networks
- Optimize ensemble weights
- Save trained models and results

**Note**: Training can take several hours depending on hardware and dataset size.

### 3. Estimate Risk

Estimate process criticality accident risk:

```bash
python estimate_risk.py
```

This will:
- Load trained models and Bayesian network
- Perform Monte Carlo risk estimation
- Save detailed results and statistics

**Note**: Requires pre-trained ensemble and Bayesian network.

## Customization

Each script can be customized by modifying the configuration variables at the top of the file. Key parameters include:

- **Data paths**: Update paths to point to your data files
- **Model architecture**: Modify layer sizes, ensemble size, etc.
- **Training parameters**: Adjust learning rate, epochs, batch size
- **Risk parameters**: Change sample sizes, cutoffs, USL values

## Data

The examples assume data is located in the R package directory structure. You may need to adjust paths if your data is located elsewhere.

Required data files:
- `facility.csv` - Facility operations data for Bayesian network
- `mcnp-dataset.csv` - Monte Carlo simulation results for training

## Output

Results are saved to the `./output/` directory by default, including:
- Trained models and checkpoints
- Training histories and plots
- Preprocessor state
- Risk estimation results
