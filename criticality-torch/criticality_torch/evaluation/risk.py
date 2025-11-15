"""Risk estimation for process criticality accidents."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import logging

if TYPE_CHECKING:
    from criticality_torch.data.bayesian_network import BayesianNetwork
    from criticality_torch.models.ensemble import NeuralEnsemble
    from criticality_torch.evaluation.predict import Predictor

logger = logging.getLogger(__name__)


@dataclass
class RiskResults:
    """Container for risk estimation results.

    Attributes:
        mean_risk: Mean estimated risk
        std_risk: Standard deviation of risk
        risk_samples: Individual risk estimates
        bn_samples: Bayesian network samples with predictions
        usl: Upper subcritical limit used
        sample_size: Sample size per iteration
        risk_pool: Number of risk iterations
    """
    mean_risk: float
    std_risk: float
    risk_samples: np.ndarray
    bn_samples: pd.DataFrame
    usl: float
    sample_size: int
    risk_pool: int


class RiskEstimator:
    """Estimator for process criticality accident risk.

    Args:
        bayesian_network: Fitted Bayesian network
        ensemble: Trained neural ensemble
        predictor: Predictor for generating keff values
    """

    def __init__(
        self,
        bayesian_network: "BayesianNetwork",
        ensemble: "NeuralEnsemble",
        predictor: Optional["Predictor"] = None
    ) -> None:
        """Initialize risk estimator.

        Args:
            bayesian_network: BN for sampling
            ensemble: Ensemble for prediction
            predictor: Optional predictor (will create if not provided)
        """
        self.bn = bayesian_network
        self.ensemble = ensemble

        if predictor is None:
            from criticality_torch.evaluation.predict import Predictor
            self.predictor = Predictor(
                bayesian_network,
                ensemble,
                ensemble.preprocessor
            )
        else:
            self.predictor = predictor

    def estimate(
        self,
        sample_size: int = 1000000,
        risk_pool: int = 100,
        usl: float = 0.95,
        keff_cutoff: float = 0.9,
        mass_cutoff: float = 100.0,
        rad_cutoff: float = 7.0,
        n_cores: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> RiskResults:
        """Estimate process criticality accident risk.

        Args:
            sample_size: Number of samples per iteration
            risk_pool: Number of risk iterations
            usl: Upper subcritical limit threshold
            keff_cutoff: Initial keff filter threshold
            mass_cutoff: Mass threshold (g)
            rad_cutoff: Radius threshold (cm)
            n_cores: Number of CPU cores for sampling
            output_dir: Optional directory to save results
            verbose: Show progress

        Returns:
            RiskResults object
        """
        # Adjust for memory constraints (max ~12 GB)
        max_samples = 500000000
        if sample_size > max_samples:
            risk_pool = int(risk_pool * sample_size / max_samples)
            sample_size = max_samples
            logger.info(
                f"Adjusted: sample_size={sample_size}, risk_pool={risk_pool}"
            )

        # Create output directory if specified
        if output_dir:
            output_path = self._create_output_dir(output_dir)
            self._save_settings(
                output_path, sample_size, risk_pool, usl,
                keff_cutoff, mass_cutoff, rad_cutoff
            )

            # Check if already computed
            results = self._load_existing_results(output_path, risk_pool)
            if results is not None:
                return results

        # Estimate risk through Monte Carlo sampling
        logger.info(f"Estimating risk over {risk_pool} iterations...")

        risk_estimates = []
        all_samples = []

        iterator = tqdm(range(risk_pool)) if verbose else range(risk_pool)

        for i in iterator:
            # Generate predictions
            samples = self.predictor.predict_from_samples(
                sample_size=sample_size,
                keff_cutoff=keff_cutoff,
                mass_cutoff=mass_cutoff,
                rad_cutoff=rad_cutoff,
                n_cores=n_cores
            )

            # Calculate risk for this iteration
            if len(samples) > 0:
                n_critical = (samples["keff"] >= usl).sum()
                risk = n_critical / sample_size
            else:
                risk = 0.0

            risk_estimates.append(risk)
            all_samples.append(samples)

            if verbose:
                iterator.set_postfix({"risk": f"{risk:.3e}"})

        # Combine all samples
        combined_samples = pd.concat(all_samples, ignore_index=True)

        # Calculate statistics
        risk_array = np.array(risk_estimates)
        mean_risk = np.mean(risk_array)
        std_risk = np.std(risk_array) if risk_pool > 1 else 0.0

        # Log results
        if mean_risk > 0:
            logger.info(f"Risk = {mean_risk:.3e}")
            if risk_pool > 1:
                logger.info(f"SD = {std_risk:.3e}")
            else:
                logger.info("SD = NA (single iteration)")
        else:
            logger.info(f"Risk < {1.0 / (risk_pool * sample_size):.3e}")

        # Create results object
        results = RiskResults(
            mean_risk=mean_risk,
            std_risk=std_risk,
            risk_samples=risk_array,
            bn_samples=combined_samples,
            usl=usl,
            sample_size=sample_size,
            risk_pool=risk_pool
        )

        # Save results
        if output_dir:
            self._save_results(output_path, results)

        return results

    def _create_output_dir(self, base_dir: str) -> Path:
        """Create timestamped output directory.

        Args:
            base_dir: Base directory

        Returns:
            Path to output directory
        """
        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        output_path = Path(base_dir) / "risk" / f"risk-{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _save_settings(
        self,
        output_path: Path,
        sample_size: int,
        risk_pool: int,
        usl: float,
        keff_cutoff: float,
        mass_cutoff: float,
        rad_cutoff: float
    ) -> None:
        """Save risk estimation settings.

        Args:
            output_path: Output directory
            sample_size: Sample size
            risk_pool: Risk pool size
            usl: USL value
            keff_cutoff: Keff cutoff
            mass_cutoff: Mass cutoff
            rad_cutoff: Radius cutoff
        """
        settings = [
            "risk settings",
            f"distribution: {self.bn.distribution}",
            f"keff cutoff: {keff_cutoff}",
            f"mass cutoff (g): {mass_cutoff}",
            f"rad cutoff (cm): {rad_cutoff}",
            f"risk pool: {risk_pool}",
            f"sample size: {sample_size}",
            f"upper subcritical limit: {usl}",
        ]

        with open(output_path / "risk-settings.txt", "w") as f:
            f.write("\n".join(settings))

    def _save_results(self, output_path: Path, results: RiskResults) -> None:
        """Save risk estimation results.

        Args:
            output_path: Output directory
            results: Risk results
        """
        # Save risk estimates
        risk_df = pd.DataFrame({"risk": results.risk_samples})
        risk_df.to_csv(output_path / "risk.csv", index=False)

        # Save BN samples with predictions
        results.bn_samples.to_csv(output_path / "bn-risk.csv", index=False)

        logger.info(f"Results saved to {output_path}")

    def _load_existing_results(
        self,
        output_path: Path,
        risk_pool: int
    ) -> Optional[RiskResults]:
        """Load existing risk results if available.

        Args:
            output_path: Output directory
            risk_pool: Expected risk pool size

        Returns:
            RiskResults if exists, None otherwise
        """
        risk_csv = output_path / "risk.csv"
        bn_csv = output_path / "bn-risk.csv"

        if not (risk_csv.exists() and bn_csv.exists()):
            return None

        # Load risk estimates
        risk_df = pd.read_csv(risk_csv)

        if len(risk_df) < risk_pool:
            return None

        # Load BN samples
        bn_samples = pd.read_csv(bn_csv)

        # Create results
        risk_array = risk_df["risk"].values[:risk_pool]
        mean_risk = np.mean(risk_array)
        std_risk = np.std(risk_array) if risk_pool > 1 else 0.0

        logger.info("Loaded existing risk results")
        logger.info(f"Risk = {mean_risk:.3e}")
        if risk_pool > 1:
            logger.info(f"SD = {std_risk:.3e}")

        # Parse settings
        settings_path = output_path / "risk-settings.txt"
        if settings_path.exists():
            with open(settings_path) as f:
                lines = f.readlines()
                usl = float([l for l in lines if "upper subcritical" in l][0].split(":")[-1])
                sample_size = int([l for l in lines if "sample size" in l][0].split(":")[-1])
        else:
            usl = 0.95
            sample_size = 1000000

        return RiskResults(
            mean_risk=mean_risk,
            std_risk=std_risk,
            risk_samples=risk_array,
            bn_samples=bn_samples,
            usl=usl,
            sample_size=sample_size,
            risk_pool=risk_pool
        )
