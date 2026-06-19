"""
Shared test fixtures for integration tests.

Provides temporary directories, test data, and environment setup
for testing the full ML pipeline.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_wine_data(temp_dir):
    """Create a small sample wine dataset for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "fixed acidity": np.random.uniform(4.0, 16.0, n_samples),
        "volatile acidity": np.random.uniform(0.1, 1.6, n_samples),
        "citric acid": np.random.uniform(0.0, 1.0, n_samples),
        "residual sugar": np.random.uniform(0.6, 15.0, n_samples),
        "chlorides": np.random.uniform(0.01, 0.35, n_samples),
        "free sulfur dioxide": np.random.uniform(1.0, 72.0, n_samples),
        "total sulfur dioxide": np.random.uniform(6.0, 289.0, n_samples),
        "density": np.random.uniform(0.990, 1.040, n_samples),
        "pH": np.random.uniform(2.7, 4.0, n_samples),
        "sulphates": np.random.uniform(0.3, 2.0, n_samples),
        "alcohol": np.random.uniform(8.0, 14.0, n_samples),
        "quality": np.random.choice([5, 6, 7], n_samples, p=[0.4, 0.4, 0.2]),
    }

    df = pd.DataFrame(data)
    data_path = temp_dir / "wine.csv"
    df.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def test_config_dir(temp_dir):
    """Create a minimal config structure for testing."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal config.yaml
    config_yaml = config_dir / "config.yaml"
    config_yaml.write_text("""
data_ingestion:
  root_dir: {root}/artifacts/data_ingestion
  source_URL: https://github.com/entbappy/Wine-Quality-Prediction/raw/main/winequality-red.csv
  local_data_file: wine.csv
  unzip_dir: {root}/artifacts/data_ingestion

data_validation:
  root_dir: {root}/artifacts/data_validation
  unzip_data_dir: {root}/artifacts/data_ingestion/wine.csv
  STATUS_FILE: {root}/artifacts/data_validation/status.txt
  all_schema:
    columns:
      fixed acidity: float64
      volatile acidity: float64
      citric acid: float64
      residual sugar: float64
      chlorides: float64
      free sulfur dioxide: float64
      total sulfur dioxide: float64
      density: float64
      pH: float64
      sulphates: float64
      alcohol: float64
      quality: int64

data_transformation:
  root_dir: {root}/artifacts/data_transformation
  data_path: {root}/artifacts/data_ingestion/wine.csv

model_trainer:
  root_dir: {root}/artifacts/model_trainer
  train_data_path: {root}/artifacts/data_transformation/train.csv
  test_data_path: {root}/artifacts/data_transformation/test.csv
  model_name: model.joblib
  alpha: 0.5
  l1_ratio: 0.5
  target_column: quality

model_evaluation:
  root_dir: {root}/artifacts/model_evaluation
  test_data_path: {root}/artifacts/data_transformation/test.csv
  model_path: {root}/artifacts/model_trainer/model.joblib
  metric_file_name: {root}/artifacts/model_evaluation/metrics.json
  model_info_path: {root}/artifacts/model_trainer/model_info.json
""".format(root=str(temp_dir)))

    # Create params.yaml
    params_yaml = temp_dir / "params.yaml"
    params_yaml.write_text("""
ElasticNet:
  alpha: 0.5
  l1_ratio: 0.5
  max_iter: 1000
""")

    # Create schema.yaml
    schema_yaml = temp_dir / "schema.yaml"
    schema_yaml.write_text("""
columns:
  fixed acidity: float64
  volatile acidity: float64
  citric acid: float64
  residual sugar: float64
  chlorides: float64
  free sulfur dioxide: float64
  total sulfur dioxide: float64
  density: float64
  pH: float64
  sulphates: float64
  alcohol: float64
  quality: int64
target_column: quality
""")

    return temp_dir


@pytest.fixture
def mock_data_source(monkeypatch, sample_wine_data):
    """Mock the data download URL to use local test data."""
    def mock_download(url, dest_path):
        import shutil
        shutil.copy(str(sample_wine_data), str(dest_path))
        return dest_path

    # We'll use this to mock the download in integration tests
    return mock_download


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables for each test."""
    monkeypatch.delenv("TRAIN_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.delenv("ENV_TAG", raising=False)
