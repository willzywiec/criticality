"""Tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
import torch
from criticality_torch.data.preprocessing import DataPreprocessor


def create_sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "mass": np.random.uniform(100, 1000, 100),
        "form": np.random.choice(["alpha", "delta"], 100),
        "mod": np.random.choice(["h2o", "none"], 100),
        "rad": np.random.uniform(5, 20, 100),
        "ref": np.random.choice(["h2o", "none"], 100),
        "thk": np.random.uniform(0, 10, 100),
        "keff": np.random.uniform(0.5, 1.2, 100),
        "sd": np.random.uniform(0.001, 0.01, 100),
    })


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    prep = DataPreprocessor(code="mcnp", test_split=0.2)

    assert prep.code == "mcnp"
    assert prep.test_split == 0.2
    assert prep.training_mean is None
    assert prep.training_std is None


def test_calculate_derived_features():
    """Test derived feature calculation."""
    prep = DataPreprocessor()
    df = create_sample_data()

    result = prep._calculate_derived_features(df)

    assert "vol" in result.columns
    assert "conc" in result.columns
    assert len(result) == len(df)


def test_one_hot_encoding():
    """Test one-hot encoding."""
    prep = DataPreprocessor()
    df = create_sample_data()

    prep.categorical_columns = ["form", "mod", "ref"]
    result = prep._one_hot_encode(df)

    # Check that categorical columns are encoded
    assert "formalpha" in result.columns or "formdelta" in result.columns
    assert "form" not in result.columns


def test_scale_and_extract():
    """Test feature scaling."""
    prep = DataPreprocessor()
    df = create_sample_data()
    df = prep._calculate_derived_features(df)
    df_encoded = prep._one_hot_encode(df)

    X, y = prep._scale_and_extract(df_encoded, fit=True)

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert prep.training_mean is not None
    assert prep.training_std is not None


def test_remove_constant_columns():
    """Test removal of constant columns."""
    prep = DataPreprocessor()
    df = pd.DataFrame({
        "constant": [1.0] * 10,
        "variable": np.random.randn(10),
        "keff": np.random.randn(10),
    })

    result = prep._remove_constant_columns(df)

    assert "constant" not in result.columns
    assert "variable" in result.columns
    assert "keff" in result.columns
