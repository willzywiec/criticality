"""Example: Estimate process criticality accident risk."""

import sys
from pathlib import Path

# Add package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from criticality_torch.data.bayesian_network import BayesianNetwork
from criticality_torch.models.ensemble import NeuralEnsemble
from criticality_torch.evaluation.risk import RiskEstimator
from criticality_torch.evaluation.predict import Predictor
from criticality_torch.data.preprocessing import DataPreprocessor
from criticality_torch.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")

def main():
    """Estimate criticality risk."""

    # Paths (adjust as needed)
    facility_csv = "../pkg/criticality/inst/extdata/facility.csv"
    ensemble_path = "./output/training/ensemble.pt"
    preprocessor_path = "./output/training/mcnp-preprocessor.pkl"
    bn_path = "./output/facility-gamma.pkl"
    output_dir = "./output"

    # Load or create Bayesian network
    print("Loading Bayesian network...")
    if Path(bn_path).exists():
        bn = BayesianNetwork.load(bn_path)
    else:
        bn = BayesianNetwork.from_csv(facility_csv, distribution="gamma")
        bn.save(bn_path)

    # Load ensemble
    print("Loading ensemble...")
    # Note: Need to know input_dim - would typically get from preprocessor
    # For this example, assuming it's been saved with preprocessor
    preprocessor = DataPreprocessor.load(preprocessor_path)
    ensemble = NeuralEnsemble.load(ensemble_path, input_dim=100)  # Adjust input_dim
    ensemble.preprocessor = preprocessor

    # Create predictor
    predictor = Predictor(bn, ensemble, preprocessor)

    # Create risk estimator
    risk_estimator = RiskEstimator(bn, ensemble, predictor)

    # Estimate risk
    print("\nEstimating risk (this may take a while)...")
    results = risk_estimator.estimate(
        sample_size=100000,  # Reduced for example
        risk_pool=10,  # Reduced for example
        usl=0.95,
        keff_cutoff=0.5,
        mass_cutoff=100.0,
        rad_cutoff=7.0,
        output_dir=output_dir,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"Risk Estimation Results")
    print(f"{'='*60}")
    print(f"Mean Risk: {results.mean_risk:.3e}")
    print(f"Std Dev:   {results.std_risk:.3e}")
    print(f"Samples:   {results.sample_size} x {results.risk_pool} iterations")
    print(f"USL:       {results.usl}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
