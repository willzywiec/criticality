"""Example: Build Bayesian network from facility data."""

import sys
from pathlib import Path

# Add package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from criticality_torch.data.bayesian_network import BayesianNetwork
from criticality_torch.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")

def main():
    """Build Bayesian network from facility CSV."""

    # Configuration
    facility_csv = "../pkg/criticality/inst/extdata/facility.csv"  # Adjust path as needed
    output_path = "./output/facility-gamma.pkl"

    # Build Bayesian network
    print("Building Bayesian network...")
    bn = BayesianNetwork.from_csv(
        facility_csv=facility_csv,
        distribution="gamma",
        save_path=output_path
    )

    print(f"\nBayesian network built successfully!")
    print(f"Operations: {bn.operations}")
    print(f"Controls: {bn.controls}")
    print(f"Forms: {bn.forms}")
    print(f"Saved to {output_path}")

    # Generate sample
    print("\nGenerating sample...")
    samples = bn.sample(n_samples=10000)
    print(f"Generated {len(samples)} samples")
    print("\nSample statistics:")
    print(samples.describe())

if __name__ == "__main__":
    main()
