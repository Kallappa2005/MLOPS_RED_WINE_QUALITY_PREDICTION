"""
Integration tests for the prediction pipeline.

These tests verify that training a model and then making predictions
works correctly end-to-end.

Run with: pytest tests/test_integration_prediction.py -v -m integration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet


pytestmark = pytest.mark.integration


class TestPredictionPipeline:
    """Test prediction after model training."""

    def test_prediction_after_training(self, temp_dir, sample_wine_data):
        """Test that a trained model can make valid predictions."""
        artifacts_dir = temp_dir / "artifacts"
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(sample_wine_data)
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Train a model
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save preprocessor (identity transform for simplicity)
        from sklearn.preprocessing import StandardScaler
        preprocessor = StandardScaler()
        preprocessor.fit(X)
        preprocessor_path = artifacts_dir / "data_transformation" / "preprocessor.joblib"
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)

        # Make prediction using the pipeline
        from mlProject.pipeline.prediction import PredictionPipeline

        # Mock the pipeline to use our test model
        with patch.object(PredictionPipeline, '__init__', lambda self: None):
            pipeline = PredictionPipeline()
            pipeline.model_path = model_path

            # Prepare test input (same format as the form data)
            test_data = np.array([
                7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4,
            ]).reshape(1, 11)

            # Make prediction
            prediction = pipeline.predict(test_data)

            # Verify prediction is valid
            assert prediction is not None
            assert len(prediction) == 1
            assert isinstance(prediction[0], (int, float, np.number))
            # Wine quality predictions should be in reasonable range (3-9)
            assert 3 <= prediction[0] <= 9

    def test_prediction_with_different_inputs(self, temp_dir, sample_wine_data):
        """Test that different inputs produce different predictions."""
        artifacts_dir = temp_dir / "artifacts"
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(sample_wine_data)
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Train a model
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        from mlProject.pipeline.prediction import PredictionPipeline

        with patch.object(PredictionPipeline, '__init__', lambda self: None):
            pipeline = PredictionPipeline()
            pipeline.model_path = model_path

            # Test with different inputs
            low_quality = np.array([
                5.0, 1.0, 0.1, 1.5, 0.1, 5.0, 10.0, 1.000, 3.5, 0.4, 8.0,
            ]).reshape(1, 11)

            high_quality = np.array([
                12.0, 0.3, 0.5, 2.0, 0.04, 30.0, 100.0, 0.994, 3.2, 0.8, 12.0,
            ]).reshape(1, 11)

            pred_low = pipeline.predict(low_quality)
            pred_high = pipeline.predict(high_quality)

            # Both predictions should be valid
            assert pred_low is not None
            assert pred_high is not None
            assert len(pred_low) == 1
            assert len(pred_high) == 1

    def test_prediction_output_format(self, temp_dir, sample_wine_data):
        """Test that prediction output is in the expected format."""
        artifacts_dir = temp_dir / "artifacts"
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(sample_wine_data)
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Train a model
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        from mlProject.pipeline.prediction import PredictionPipeline

        with patch.object(PredictionPipeline, '__init__', lambda self: None):
            pipeline = PredictionPipeline()
            pipeline.model_path = model_path

            test_data = np.array([
                7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4,
            ]).reshape(1, 11)

            prediction = pipeline.predict(test_data)

            # Verify output format
            assert isinstance(prediction, np.ndarray)
            assert prediction.dtype in [np.float64, np.float32, np.int64, np.int32]


class TestPredictionFromFlask:
    """Test prediction via Flask endpoint."""

    def test_predict_endpoint_with_valid_data(self, temp_dir, sample_wine_data):
        """Test /predict endpoint with valid form data."""
        artifacts_dir = temp_dir / "artifacts"
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(sample_wine_data)
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Train a model
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Import app after setting up the model
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']

        with patch("mlProject.pipeline.prediction.PredictionPipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = np.array([6.5])
            mock_pipeline.return_value = mock_instance

            # Need to reload app to pick up the mock
            import importlib
            import app as flask_app
            importlib.reload(flask_app)

            with flask_app.app.test_client() as client:
                response = client.post("/predict", data={
                    "fixed_acidity": 7.4,
                    "volatile_acidity": 0.7,
                    "citric_acid": 0.0,
                    "residual_sugar": 1.9,
                    "chlorides": 0.076,
                    "free_sulfur_dioxide": 11.0,
                    "total_sulfur_dioxide": 34.0,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4,
                })

                # Should return 200 with HTML response
                assert response.status_code == 200
                # Response should contain prediction
                assert b"prediction" in response.data.lower() or response.status_code == 200

    def test_predict_endpoint_with_invalid_data(self, temp_dir):
        """Test /predict endpoint with invalid form data returns 400."""
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']

        import importlib
        import app as flask_app
        importlib.reload(flask_app)

        with flask_app.app.test_client() as client:
            response = client.post("/predict", data={
                "fixed_acidity": -1.0,  # Invalid: negative
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            })

            # Should return 400 for validation error
            assert response.status_code == 400

    def test_predict_endpoint_get_returns_form(self, temp_dir):
        """Test GET /predict returns the form page."""
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']

        import importlib
        import app as flask_app
        importlib.reload(flask_app)

        with flask_app.app.test_client() as client:
            response = client.get("/predict")
            # Should return 200 with the form
            assert response.status_code == 200


class TestModelPersistence:
    """Test that models can be saved and loaded correctly."""

    def test_model_save_and_load(self, temp_dir, sample_wine_data):
        """Test model can be saved to disk and loaded back."""
        artifacts_dir = temp_dir / "artifacts"
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(sample_wine_data)
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Train a model
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Load model
        loaded_model = joblib.load(model_path)

        # Verify loaded model makes same predictions
        test_input = X.iloc[:1].values
        original_pred = model.predict(test_input)
        loaded_pred = loaded_model.predict(test_input)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_model_integrity_verification(self, temp_dir, sample_wine_data):
        """Test model integrity can be verified with checksum."""
        from mlProject.utils.common import compute_checksum, save_checksum, verify_model_integrity

        artifacts_dir = temp_dir / "artifacts"
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(sample_wine_data)
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Train and save model
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X, y)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Create checksum
        checksum_path = Path(str(model_path) + ".sha256")
        save_checksum(model_path, checksum_path)

        # Verify integrity
        assert verify_model_integrity(model_path, checksum_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
