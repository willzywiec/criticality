"""Example: Train neural network ensemble for keff prediction."""

import sys
from pathlib import Path

# Add package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from criticality_torch.models.ensemble import NeuralEnsemble
from criticality_torch.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")

def main():
    """Train ensemble on MCNP dataset."""

    # Configuration
    data_path = "../pkg/criticality/inst/extdata/mcnp-dataset.csv"  # Adjust path as needed
    output_dir = "./output/training"

    # Create ensemble
    ensemble = NeuralEnsemble(
        layers="8192-256-256-256-256-16",
        ensemble_size=5,
        epochs=1500,
        batch_size=8192,
        loss="sse",
        optimizer_name="adamax",
        learning_rate=0.00075,
        val_split=0.2
    )

    # Train
    print("Starting ensemble training...")
    ensemble.train(
        csv_path=data_path,
        code="mcnp",
        output_dir=output_dir,
        verbose=True
    )

    print(f"\nTraining complete!")
    print(f"Results saved to {output_dir}")
    print(f"Ensemble test MAE: {ensemble.ensemble_test_mae:.6f}")

if __name__ == "__main__":
    main()
