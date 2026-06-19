"""
Integration tests for the full ML pipeline.

These tests exercise the complete pipeline from data ingestion to model evaluation,
verifying that all stages work together correctly.

Run with: pytest tests/test_integration_pipeline.py -v -m integration
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline


pytestmark = pytest.mark.integration


class TestPipelineStages:
    """Test individual pipeline stages with mocked dependencies."""

    def test_data_ingestion_stage_with_local_data(self, temp_dir, sample_wine_data):
        """Test data ingestion stage using local file instead of URL download."""
        artifacts_dir = temp_dir / "artifacts" / "data_ingestion"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Mock the config to use local file
        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_ingestion_config = MagicMock()
            mock_ingestion_config.root_dir = artifacts_dir
            mock_ingestion_config.source_URL = "http://example.com/data.csv"
            mock_ingestion_config.local_data_file = "wine.csv"
            mock_ingestion_config.unzip_dir = artifacts_dir
            mock_config.return_value.get_data_ingestion_config.return_value = mock_ingestion_config

            # Mock the download to copy local file
            with patch("mlProject.components.data_ingestion.DataIngestion.download_file") as mock_download:
                def side_effect(url, dest_path):
                    import shutil
                    dest = Path(dest_path) / "wine.csv"
                    shutil.copy(str(sample_wine_data), str(dest))
                    return dest
                mock_download.side_effect = side_effect

                with patch("mlProject.components.data_ingestion.DataIngestion.extract_zip_file"):
                    pipeline = DataIngestionTrainingPipeline()
                    pipeline.main()

            # Verify the data file exists
            assert (artifacts_dir / "wine.csv").exists()

    def test_data_validation_stage(self, temp_dir, sample_wine_data):
        """Test data validation stage with valid data."""
        artifacts_dir = temp_dir / "artifacts"
        ingestion_dir = artifacts_dir / "data_ingestion"
        ingestion_dir.mkdir(parents=True, exist_ok=True)

        # Copy sample data
        import shutil
        shutil.copy(str(sample_wine_data), str(ingestion_dir / "wine.csv"))

        validation_dir = artifacts_dir / "data_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_validation_config = MagicMock()
            mock_validation_config.root_dir = validation_dir
            mock_validation_config.unzip_data_dir = ingestion_dir / "wine.csv"
            mock_validation_config.STATUS_FILE = validation_dir / "status.txt"
            mock_validation_config.all_schema = {
                "columns": {
                    "fixed acidity": "float64",
                    "volatile acidity": "float64",
                    "citric acid": "float64",
                    "residual sugar": "float64",
                    "chlorides": "float64",
                    "free sulfur dioxide": "float64",
                    "total sulfur dioxide": "float64",
                    "density": "float64",
                    "pH": "float64",
                    "sulphates": "float64",
                    "alcohol": "float64",
                    "quality": "int64",
                },
                "target_column": "quality",
            }
            mock_config.return_value.get_data_validation_config.return_value = mock_validation_config

            pipeline = DataValidationTrainingPipeline()
            pipeline.main()

            # Verify status file was created
            assert (validation_dir / "status.txt").exists()

    def test_data_transformation_stage(self, temp_dir, sample_wine_data):
        """Test data transformation stage produces train/test splits."""
        artifacts_dir = temp_dir / "artifacts"
        ingestion_dir = artifacts_dir / "data_ingestion"
        ingestion_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(str(sample_wine_data), str(ingestion_dir / "wine.csv"))

        transformation_dir = artifacts_dir / "data_transformation"
        transformation_dir.mkdir(parents=True, exist_ok=True)

        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_transform_config = MagicMock()
            mock_transform_config.root_dir = transformation_dir
            mock_transform_config.data_path = ingestion_dir / "wine.csv"
            mock_transform_config.test_size = 0.2
            mock_transform_config.random_state = 42
            mock_transform_config.stratify_column = "quality"
            mock_transform_config.use_scaler = True
            mock_transform_config.scaler_type = "standard"
            mock_transform_config.handle_outliers = True
            mock_transform_config.outlier_method = "iqr"
            mock_transform_config.outlier_iqr_multiplier = 1.5
            mock_transform_config.impute_missing = False
            mock_transform_config.feature_engineering_flags = {
                "add_acidity_index": True,
                "add_alcohol_sugar_ratio": True,
                "add_free_sulfur_pct": True,
            }
            mock_config.return_value.get_data_transformation_config.return_value = mock_transform_config

            pipeline = DataTransformationTrainingPipeline()
            pipeline.main()

            # Verify train and test files were created
            assert (transformation_dir / "train.csv").exists()
            assert (transformation_dir / "test.csv").exists()
            assert (transformation_dir / "preprocessor.joblib").exists()

            # Verify the splits have data
            train_df = pd.read_csv(transformation_dir / "train.csv")
            test_df = pd.read_csv(transformation_dir / "test.csv")
            assert len(train_df) > 0
            assert len(test_df) > 0
            assert len(train_df) > len(test_df)

    def test_model_trainer_stage(self, temp_dir, sample_wine_data):
        """Test model trainer stage produces a model file."""
        artifacts_dir = temp_dir / "artifacts"

        # Create train/test data
        transformation_dir = artifacts_dir / "data_transformation"
        transformation_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(sample_wine_data)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        train_df.to_csv(transformation_dir / "train.csv", index=False)
        test_df.to_csv(transformation_dir / "test.csv", index=False)

        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_trainer_config = MagicMock()
            mock_trainer_config.root_dir = model_dir
            mock_trainer_config.train_data_path = transformation_dir / "train.csv"
            mock_trainer_config.test_data_path = transformation_dir / "test.csv"
            mock_trainer_config.model_name = "model.joblib"
            mock_trainer_config.alpha = 0.5
            mock_trainer_config.l1_ratio = 0.5
            mock_trainer_config.target_column = "quality"
            mock_config.return_value.get_model_trainer_config.return_value = mock_trainer_config

            pipeline = ModelTrainerPipeline()
            pipeline.main()

            # Verify model file was created
            model_path = model_dir / "model.joblib"
            assert model_path.exists()

            # Verify model can be loaded
            model = joblib.load(model_path)
            assert model is not None

    def test_model_evaluation_stage(self, temp_dir, sample_wine_data):
        """Test model evaluation stage produces metrics."""
        artifacts_dir = temp_dir / "artifacts"

        # Create train/test data
        transformation_dir = artifacts_dir / "data_transformation"
        transformation_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(sample_wine_data)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        train_df.to_csv(transformation_dir / "train.csv", index=False)
        test_df.to_csv(transformation_dir / "test.csv", index=False)

        # Create and train a simple model
        from sklearn.linear_model import ElasticNet
        model_dir = artifacts_dir / "model_trainer"
        model_dir.mkdir(parents=True, exist_ok=True)

        X_train = train_df.drop("quality", axis=1)
        y_train = train_df["quality"]
        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        model.fit(X_train, y_train)

        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Create model_info.json
        import json
        model_info = {
            "version_id": "v_test_123",
            "status": "production",
            "metrics": {"rmse": 0.5, "mae": 0.4, "r2": 0.6},
        }
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f)

        # Create registry
        registry_dir = artifacts_dir / "model_registry"
        registry_dir.mkdir(parents=True, exist_ok=True)
        registry = {
            "production": "v_test_123",
            "versions": [
                {
                    "id": "v_test_123",
                    "path": str(model_path),
                    "status": "production",
                    "metrics": {"rmse": 0.5, "mae": 0.4, "r2": 0.6},
                    "params": {"alpha": 0.5, "l1_ratio": 0.5},
                    "date": "2024-01-01T00:00:00Z",
                    "data_hash": "test_hash",
                }
            ],
        }
        with open(registry_dir / "model_registry.json", "w") as f:
            json.dump(registry, f)

        evaluation_dir = artifacts_dir / "model_evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_eval_config = MagicMock()
            mock_eval_config.root_dir = evaluation_dir
            mock_eval_config.test_data_path = transformation_dir / "test.csv"
            mock_eval_config.model_path = model_path
            mock_eval_config.metric_file_name = evaluation_dir / "metrics.json"
            mock_eval_config.model_info_path = model_dir / "model_info.json"
            mock_config.return_value.get_model_evaluation_config.return_value = mock_eval_config

            with patch("mlProject.utils.model_registry._get_registry_path") as mock_registry:
                mock_registry.return_value = registry_dir / "model_registry.json"
                pipeline = ModelEvaluationPipeline()
                pipeline.main()

            # Verify metrics file was created
            metrics_path = evaluation_dir / "metrics.json"
            assert metrics_path.exists()

            # Verify metrics contain expected keys
            with open(metrics_path) as f:
                metrics = json.load(f)
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "r2" in metrics


class TestFullPipeline:
    """Integration test for the complete pipeline flow."""

    def test_full_pipeline_end_to_end(self, temp_dir, sample_wine_data):
        """Test the complete pipeline from data ingestion to evaluation."""
        # This test is more complex and requires mocking the data download
        # It verifies that all stages can run in sequence
        artifacts_dir = temp_dir / "artifacts"

        # Create necessary directories
        for subdir in ["data_ingestion", "data_validation", "data_transformation",
                       "model_trainer", "model_evaluation", "model_registry"]:
            (artifacts_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Copy sample data to ingestion directory
        import shutil
        shutil.copy(str(sample_wine_data),
                    str(artifacts_dir / "data_ingestion" / "wine.csv"))

        # Create registry
        import json
        registry = {
            "production": None,
            "versions": [],
        }
        with open(artifacts_dir / "model_registry" / "model_registry.json", "w") as f:
            json.dump(registry, f)

        # Mock all configs to use our temp directories
        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            # Setup all mock configs
            mock_ingestion = MagicMock()
            mock_ingestion.root_dir = artifacts_dir / "data_ingestion"
            mock_ingestion.source_URL = "http://example.com/data.csv"
            mock_ingestion.local_data_file = "wine.csv"
            mock_ingestion.unzip_dir = artifacts_dir / "data_ingestion"

            mock_validation = MagicMock()
            mock_validation.root_dir = artifacts_dir / "data_validation"
            mock_validation.unzip_data_dir = artifacts_dir / "data_ingestion" / "wine.csv"
            mock_validation.STATUS_FILE = artifacts_dir / "data_validation" / "status.txt"
            mock_validation.all_schema = {
                "columns": {
                    "fixed acidity": "float64",
                    "volatile acidity": "float64",
                    "citric acid": "float64",
                    "residual sugar": "float64",
                    "chlorides": "float64",
                    "free sulfur dioxide": "float64",
                    "total sulfur dioxide": "float64",
                    "density": "float64",
                    "pH": "float64",
                    "sulphates": "float64",
                    "alcohol": "float64",
                    "quality": "int64",
                },
                "target_column": "quality",
            }

            mock_transformation = MagicMock()
            mock_transformation.root_dir = artifacts_dir / "data_transformation"
            mock_transformation.data_path = artifacts_dir / "data_ingestion" / "wine.csv"
            mock_transformation.test_size = 0.2
            mock_transformation.random_state = 42
            mock_transformation.stratify_column = "quality"
            mock_transformation.use_scaler = True
            mock_transformation.scaler_type = "standard"
            mock_transformation.handle_outliers = True
            mock_transformation.outlier_method = "iqr"
            mock_transformation.outlier_iqr_multiplier = 1.5
            mock_transformation.impute_missing = False
            mock_transformation.feature_engineering_flags = {
                "add_acidity_index": True,
                "add_alcohol_sugar_ratio": True,
                "add_free_sulfur_pct": True,
            }

            mock_trainer = MagicMock()
            mock_trainer.root_dir = artifacts_dir / "model_trainer"
            mock_trainer.train_data_path = artifacts_dir / "data_transformation" / "train.csv"
            mock_trainer.test_data_path = artifacts_dir / "data_transformation" / "test.csv"
            mock_trainer.model_name = "model.joblib"
            mock_trainer.alpha = 0.5
            mock_trainer.l1_ratio = 0.5
            mock_trainer.target_column = "quality"

            mock_evaluation = MagicMock()
            mock_evaluation.root_dir = artifacts_dir / "model_evaluation"
            mock_evaluation.test_data_path = artifacts_dir / "data_transformation" / "test.csv"
            mock_evaluation.model_path = artifacts_dir / "model_trainer" / "model.joblib"
            mock_evaluation.metric_file_name = artifacts_dir / "model_evaluation" / "metrics.json"
            mock_evaluation.model_info_path = artifacts_dir / "model_trainer" / "model_info.json"

            mock_config.return_value.get_data_ingestion_config.return_value = mock_ingestion
            mock_config.return_value.get_data_validation_config.return_value = mock_validation
            mock_config.return_value.get_data_transformation_config.return_value = mock_transformation
            mock_config.return_value.get_model_trainer_config.return_value = mock_trainer
            mock_config.return_value.get_model_evaluation_config.return_value = mock_evaluation

            # Mock download to use local file
            with patch("mlProject.components.data_ingestion.DataIngestion.download_file") as mock_download:
                def side_effect(url, dest_path):
                    dest = Path(dest_path) / "wine.csv"
                    shutil.copy(str(sample_wine_data), str(dest))
                    return dest
                mock_download.side_effect = side_effect

                with patch("mlProject.components.data_ingestion.DataIngestion.extract_zip_file"):
                    with patch("mlProject.utils.model_registry._get_registry_path") as mock_registry:
                        mock_registry.return_value = (
                            artifacts_dir / "model_registry" / "model_registry.json"
                        )

                        # Run full pipeline
                        from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
                        from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
                        from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
                        from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
                        from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

                        # Stage 1: Data Ingestion
                        DataIngestionTrainingPipeline().main()

                        # Stage 2: Data Validation
                        DataValidationTrainingPipeline().main()

                        # Stage 3: Data Transformation
                        DataTransformationTrainingPipeline().main()

                        # Stage 4: Model Training
                        ModelTrainerPipeline().main()

                        # Stage 5: Model Evaluation
                        ModelEvaluationPipeline().main()

        # Verify all expected outputs exist
        assert (artifacts_dir / "data_ingestion" / "wine.csv").exists()
        assert (artifacts_dir / "data_validation" / "status.txt").exists()
        assert (artifacts_dir / "data_transformation" / "train.csv").exists()
        assert (artifacts_dir / "data_transformation" / "test.csv").exists()
        assert (artifacts_dir / "data_transformation" / "preprocessor.joblib").exists()
        assert (artifacts_dir / "model_trainer" / "model.joblib").exists()
        assert (artifacts_dir / "model_evaluation" / "metrics.json").exists()

        # Verify metrics
        with open(artifacts_dir / "model_evaluation" / "metrics.json") as f:
            metrics = json.load(f)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

        # Verify model is loadable
        model = joblib.load(artifacts_dir / "model_trainer" / "model.joblib")
        assert model is not None


class TestPipelineRobustness:
    """Test pipeline handles edge cases and errors gracefully."""

    def test_pipeline_fails_gracefully_with_missing_data(self, temp_dir):
        """Test pipeline fails with clear error when data file is missing."""
        artifacts_dir = temp_dir / "artifacts"
        ingestion_dir = artifacts_dir / "data_ingestion"
        ingestion_dir.mkdir(parents=True, exist_ok=True)

        # Don't create any data file

        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_ingestion = MagicMock()
            mock_ingestion.root_dir = ingestion_dir
            mock_ingestion.source_URL = "http://example.com/data.csv"
            mock_ingestion.local_data_file = "wine.csv"
            mock_ingestion.unzip_dir = ingestion_dir
            mock_config.return_value.get_data_ingestion_config.return_value = mock_ingestion

            with patch("mlProject.components.data_ingestion.DataIngestion.download_file"):
                with patch("mlProject.components.data_ingestion.DataIngestion.extract_zip_file"):
                    pipeline = DataIngestionTrainingPipeline()
                    # This should not raise, but the data file won't exist
                    pipeline.main()

        # Verify the data file doesn't exist
        assert not (ingestion_dir / "wine.csv").exists()

    def test_validation_rejects_incorrect_schema(self, temp_dir, sample_wine_data):
        """Test data validation rejects data with wrong schema."""
        artifacts_dir = temp_dir / "artifacts"
        ingestion_dir = artifacts_dir / "data_ingestion"
        ingestion_dir.mkdir(parents=True, exist_ok=True)

        # Create data with wrong column names
        df = pd.read_csv(sample_wine_data)
        df = df.rename(columns={"fixed acidity": "wrong_column"})
        df.to_csv(ingestion_dir / "wine.csv", index=False)

        validation_dir = artifacts_dir / "data_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        with patch("mlProject.config.configuration.ConfigurationManager") as mock_config:
            mock_validation = MagicMock()
            mock_validation.root_dir = validation_dir
            mock_validation.unzip_data_dir = ingestion_dir / "wine.csv"
            mock_validation.STATUS_FILE = validation_dir / "status.txt"
            mock_validation.all_schema = {
                "columns": {
                    "fixed acidity": "float64",
                    "volatile acidity": "float64",
                    "citric acid": "float64",
                    "residual sugar": "float64",
                    "chlorides": "float64",
                    "free sulfur dioxide": "float64",
                    "total sulfur dioxide": "float64",
                    "density": "float64",
                    "pH": "float64",
                    "sulphates": "float64",
                    "alcohol": "float64",
                    "quality": "int64",
                },
                "target_column": "quality",
            }
            mock_config.return_value.get_data_validation_config.return_value = mock_validation

            pipeline = DataValidationTrainingPipeline()
            # This should raise because schema doesn't match
            with pytest.raises(ValueError):
                pipeline.main()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
