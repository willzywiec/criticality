"""Bayesian network for modeling facility operations and criticality parameters."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)


class BayesianNetwork:
    """Bayesian network for criticality safety modeling.

    Models relationships between:
    - Operations (op)
    - Controls (ctrl)
    - Physical parameters (mass, form, mod, rad, ref, thk)

    Args:
        distribution: Probability distribution for continuous variables
                     ('gamma', 'normal', 'log-normal', 'weibull', 'gev')
    """

    def __init__(self, distribution: str = "gamma") -> None:
        """Initialize Bayesian network.

        Args:
            distribution: Distribution type for continuous parameters
        """
        self.distribution = distribution

        # Categorical parameters
        self.operations: List[str] = []
        self.controls: List[str] = []
        self.forms: List[str] = []
        self.moderators: List[str] = []
        self.reflectors: List[str] = []

        # Discrete parameter ranges
        self.mass_range = np.arange(0, 4001, 1)
        self.rad_range = np.arange(0, 18.25, 0.25) * 2.54  # inches to cm
        self.thk_range = np.arange(0, 11.25, 0.25) * 2.54  # inches to cm

        # Conditional probability tables
        self.op_cpt: Optional[np.ndarray] = None
        self.ctrl_cpt: Optional[np.ndarray] = None
        self.mass_cpt: Optional[np.ndarray] = None
        self.form_cpt: Optional[np.ndarray] = None
        self.mod_cpt: Optional[np.ndarray] = None
        self.rad_cpt: Optional[np.ndarray] = None
        self.ref_cpt: Optional[np.ndarray] = None
        self.thk_cpt: Optional[np.ndarray] = None

        # Network structure
        self.nodes = ["op", "ctrl", "mass", "form", "mod", "rad", "ref", "thk"]

    @classmethod
    def from_csv(
        cls,
        facility_csv: str,
        distribution: str = "gamma",
        save_path: Optional[str] = None
    ) -> "BayesianNetwork":
        """Build Bayesian network from facility data CSV.

        Args:
            facility_csv: Path to facility data CSV
            distribution: Distribution type
            save_path: Optional path to save fitted network

        Returns:
            Fitted BayesianNetwork
        """
        bn = cls(distribution=distribution)
        facility_data = pd.read_csv(facility_csv)
        bn.fit(facility_data)

        if save_path:
            bn.save(save_path)

        return bn

    def fit(self, facility_data: pd.DataFrame) -> None:
        """Fit Bayesian network to facility data.

        Args:
            facility_data: Facility operations dataframe
        """
        logger.info(f"Fitting Bayesian network with {self.distribution} distribution")

        # Extract categorical parameters
        self.operations = sorted(facility_data["op"].unique())
        self.controls = sorted(facility_data["ctrl"].unique())
        self.forms = sorted(facility_data["form"].unique())
        self.moderators = sorted(facility_data["mod"].unique())
        self.reflectors = sorted(facility_data["ref"].unique())

        # Build operation probabilities
        self.op_cpt = self._fit_operation(facility_data)

        # Build control probabilities (conditional on operation)
        self.ctrl_cpt = self._fit_control(facility_data)

        # Build parameter probabilities (conditional on operation and control)
        self.mass_cpt = self._fit_parameter(
            facility_data, "mass", self.mass_range, is_categorical=False
        )
        self.form_cpt = self._fit_parameter(
            facility_data, "form", self.forms, is_categorical=True
        )
        self.mod_cpt = self._fit_parameter(
            facility_data, "mod", self.moderators, is_categorical=True
        )
        self.rad_cpt = self._fit_parameter(
            facility_data, "rad", self.rad_range, is_categorical=False
        )
        self.ref_cpt = self._fit_parameter(
            facility_data, "ref", self.reflectors, is_categorical=True
        )
        self.thk_cpt = self._fit_parameter(
            facility_data, "thk", self.thk_range, is_categorical=False
        )

        logger.info("Bayesian network fitting complete")

    def _fit_operation(self, data: pd.DataFrame) -> np.ndarray:
        """Fit operation probabilities.

        Args:
            data: Facility data

        Returns:
            Operation probability array
        """
        probs = []
        for op in self.operations:
            prob = (data["op"] == op).sum() / len(data)
            probs.append(prob)

        return np.array(probs)

    def _fit_control(self, data: pd.DataFrame) -> np.ndarray:
        """Fit control probabilities conditional on operation.

        Args:
            data: Facility data

        Returns:
            Control probability matrix (controls x operations)
        """
        probs = np.zeros((len(self.controls), len(self.operations)))

        for i, op in enumerate(self.operations):
            op_data = data[data["op"] == op]
            if len(op_data) == 0:
                # Uniform distribution if no data
                probs[:, i] = 1.0 / len(self.controls)
            else:
                for j, ctrl in enumerate(self.controls):
                    probs[j, i] = (op_data["ctrl"] == ctrl).sum() / len(op_data)

        return probs

    def _fit_parameter(
        self,
        data: pd.DataFrame,
        param_name: str,
        param_values: Union[List[str], np.ndarray],
        is_categorical: bool
    ) -> np.ndarray:
        """Fit parameter probabilities conditional on operation and control.

        Args:
            data: Facility data
            param_name: Parameter name
            param_values: Possible parameter values
            is_categorical: Whether parameter is categorical

        Returns:
            Parameter probability array (values x controls x operations)
        """
        n_values = len(param_values)
        n_controls = len(self.controls)
        n_operations = len(self.operations)

        probs = np.zeros((n_values, n_controls, n_operations))

        for i, op in enumerate(self.operations):
            for j, ctrl in enumerate(self.controls):
                # Filter data
                mask = (data["op"] == op) & (data["ctrl"] == ctrl)
                filtered_data = data[mask][param_name]

                if len(filtered_data) == 0:
                    # No data - assign uniform probability
                    probs[:, j, i] = 1.0 / n_values
                elif is_categorical:
                    # Categorical parameter - count frequencies
                    probs[:, j, i] = self._fit_categorical(
                        filtered_data, param_values
                    )
                else:
                    # Continuous parameter - fit distribution
                    probs[:, j, i] = self._fit_continuous(
                        filtered_data, param_values
                    )

        return probs

    def _fit_categorical(
        self,
        data: pd.Series,
        categories: List[str]
    ) -> np.ndarray:
        """Fit categorical distribution.

        Args:
            data: Observed values
            categories: All possible categories

        Returns:
            Probability array
        """
        probs = np.zeros(len(categories))

        for i, cat in enumerate(categories):
            probs[i] = (data == cat).sum() / len(data)

        return probs

    def _fit_continuous(
        self,
        data: pd.Series,
        values: np.ndarray
    ) -> np.ndarray:
        """Fit continuous distribution.

        Args:
            data: Observed values
            values: Discretized value range

        Returns:
            Probability density array
        """
        data = data.values

        # Handle single unique value case
        if len(np.unique(data)) == 1:
            # Add small perturbation to avoid fitting errors
            data = np.append(data, data[0] + 1e-6)

        try:
            if self.distribution == "gamma":
                # Fit gamma distribution
                shape, loc, scale = stats.gamma.fit(data, floc=0)
                pdf = stats.gamma.pdf(values, shape, loc, scale)

            elif self.distribution == "normal":
                # Fit normal distribution
                mean, std = stats.norm.fit(data)
                pdf = stats.norm.pdf(values, mean, std)

            elif self.distribution == "log-normal":
                # Fit log-normal distribution
                shape, loc, scale = stats.lognorm.fit(data, floc=0)
                pdf = stats.lognorm.pdf(values, shape, loc, scale)

            elif self.distribution == "weibull":
                # Fit Weibull distribution
                shape, loc, scale = stats.weibull_min.fit(data, floc=0)
                pdf = stats.weibull_min.pdf(values, shape, loc, scale)

            elif self.distribution == "gev":
                # Fit generalized extreme value distribution
                shape, loc, scale = stats.genextreme.fit(data)
                pdf = stats.genextreme.pdf(values, shape, loc, scale)

            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")

            # Normalize to sum to 1
            pdf = pdf / pdf.sum()

            return pdf

        except Exception as e:
            logger.warning(f"Distribution fitting failed: {e}, using uniform")
            return np.ones(len(values)) / len(values)

    def sample(
        self,
        n_samples: int = 1000000,
        mass_cutoff: float = 100.0,
        rad_cutoff: float = 7.0,
        n_cores: Optional[int] = None
    ) -> pd.DataFrame:
        """Sample from the Bayesian network.

        Args:
            n_samples: Number of samples to generate
            mass_cutoff: Minimum mass threshold (g)
            rad_cutoff: Minimum radius threshold (cm)
            n_cores: Number of CPU cores (default: half of available)

        Returns:
            Dataframe of samples
        """
        if n_cores is None:
            n_cores = max(1, cpu_count() // 2)

        logger.info(f"Generating {n_samples} BN samples using {n_cores} cores")

        # Sample all parameters
        samples = {
            "op": self._sample_categorical(self.op_cpt, self.operations, n_samples),
        }

        # Sample control conditional on operation
        samples["ctrl"] = self._sample_conditional_categorical(
            self.ctrl_cpt, self.controls, samples["op"]
        )

        # Sample other parameters conditional on operation and control
        samples["mass"] = self._sample_conditional_continuous(
            self.mass_cpt, self.mass_range, samples["op"], samples["ctrl"]
        )
        samples["form"] = self._sample_conditional_categorical(
            self.form_cpt, self.forms, samples["op"], samples["ctrl"]
        )
        samples["mod"] = self._sample_conditional_categorical(
            self.mod_cpt, self.moderators, samples["op"], samples["ctrl"]
        )
        samples["rad"] = self._sample_conditional_continuous(
            self.rad_cpt, self.rad_range, samples["op"], samples["ctrl"]
        )
        samples["ref"] = self._sample_conditional_categorical(
            self.ref_cpt, self.reflectors, samples["op"], samples["ctrl"]
        )
        samples["thk"] = self._sample_conditional_continuous(
            self.thk_cpt, self.thk_range, samples["op"], samples["ctrl"]
        )

        # Create dataframe
        df = pd.DataFrame(samples)

        # Apply cutoffs
        df = df[(df["mass"] > mass_cutoff) & (df["rad"] > rad_cutoff)]

        logger.info(f"BN samples generated ({len(df)} after filtering)")

        return df

    def _sample_categorical(
        self,
        cpt: np.ndarray,
        categories: List[str],
        n_samples: int
    ) -> np.ndarray:
        """Sample from categorical distribution.

        Args:
            cpt: Conditional probability table
            categories: Category names
            n_samples: Number of samples

        Returns:
            Sampled categories
        """
        indices = np.random.choice(len(categories), size=n_samples, p=cpt)
        return np.array([categories[i] for i in indices])

    def _sample_conditional_categorical(
        self,
        cpt: np.ndarray,
        categories: List[str],
        parent1: np.ndarray,
        parent2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sample from conditional categorical distribution.

        Args:
            cpt: Conditional probability table
            categories: Category names
            parent1: Parent variable values (operation)
            parent2: Optional second parent (control)

        Returns:
            Sampled categories
        """
        n_samples = len(parent1)
        samples = np.empty(n_samples, dtype=object)

        for i in range(n_samples):
            # Get parent indices
            op_idx = self.operations.index(parent1[i])

            if parent2 is not None:
                ctrl_idx = self.controls.index(parent2[i])
                probs = cpt[:, ctrl_idx, op_idx]
            else:
                probs = cpt[:, op_idx]

            # Sample
            cat_idx = np.random.choice(len(categories), p=probs)
            samples[i] = categories[cat_idx]

        return samples

    def _sample_conditional_continuous(
        self,
        cpt: np.ndarray,
        values: np.ndarray,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> np.ndarray:
        """Sample from conditional continuous distribution.

        Args:
            cpt: Conditional probability table
            values: Possible values
            parent1: First parent (operation)
            parent2: Second parent (control)

        Returns:
            Sampled values
        """
        n_samples = len(parent1)
        samples = np.empty(n_samples)

        for i in range(n_samples):
            op_idx = self.operations.index(parent1[i])
            ctrl_idx = self.controls.index(parent2[i])
            probs = cpt[:, ctrl_idx, op_idx]

            # Sample value
            val_idx = np.random.choice(len(values), p=probs)
            samples[i] = values[val_idx]

        return samples

    def save(self, path: str) -> None:
        """Save Bayesian network to file.

        Args:
            path: Save path
        """
        state = {
            "distribution": self.distribution,
            "operations": self.operations,
            "controls": self.controls,
            "forms": self.forms,
            "moderators": self.moderators,
            "reflectors": self.reflectors,
            "mass_range": self.mass_range,
            "rad_range": self.rad_range,
            "thk_range": self.thk_range,
            "op_cpt": self.op_cpt,
            "ctrl_cpt": self.ctrl_cpt,
            "mass_cpt": self.mass_cpt,
            "form_cpt": self.form_cpt,
            "mod_cpt": self.mod_cpt,
            "rad_cpt": self.rad_cpt,
            "ref_cpt": self.ref_cpt,
            "thk_cpt": self.thk_cpt,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Bayesian network saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BayesianNetwork":
        """Load Bayesian network from file.

        Args:
            path: Path to saved network

        Returns:
            Loaded BayesianNetwork
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        bn = cls(distribution=state["distribution"])
        bn.operations = state["operations"]
        bn.controls = state["controls"]
        bn.forms = state["forms"]
        bn.moderators = state["moderators"]
        bn.reflectors = state["reflectors"]
        bn.mass_range = state["mass_range"]
        bn.rad_range = state["rad_range"]
        bn.thk_range = state["thk_range"]
        bn.op_cpt = state["op_cpt"]
        bn.ctrl_cpt = state["ctrl_cpt"]
        bn.mass_cpt = state["mass_cpt"]
        bn.form_cpt = state["form_cpt"]
        bn.mod_cpt = state["mod_cpt"]
        bn.rad_cpt = state["rad_cpt"]
        bn.ref_cpt = state["ref_cpt"]
        bn.thk_cpt = state["thk_cpt"]

        logger.info(f"Bayesian network loaded from {path}")
        return bn
