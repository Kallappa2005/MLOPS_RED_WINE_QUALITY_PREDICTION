import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject.components.xai_explainer import XAIExplainer
from mlProject.components.monitoring import PredictionLogger, DriftDetector
from mlProject.components.experiment_tracker import get_mlflow_runs
from mlProject.components.analytics import get_analytics_summary, generate_pdf_report
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def dummy_model_and_data(tmp_path):
    features = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    data = np.random.rand(20, len(features))
    df = pd.DataFrame(data, columns=features)
    df["quality"] = np.random.randint(3, 9, 20)
    
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)
    
    model = ElasticNet()
    model.fit(df[features].values, df["quality"].values)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipeline.fit(df[features].values, df["quality"].values)
    
    return pipeline, str(train_path), features

def test_xai_explainer(dummy_model_and_data):
    model, train_path, features = dummy_model_and_data
    explainer = XAIExplainer(model, training_data_path=train_path)
    
    sample_input = {feat: 0.5 for feat in features}
    explanation = explainer.explain_instance(sample_input)
    assert "prediction" in explanation
    assert "base_value" in explanation
    assert len(explanation["contributions"]) == len(features)
    
    global_importance = explainer.get_global_importance()
    assert "importances" in global_importance
    assert len(global_importance["importances"]) == len(features)

def test_monitoring_drift_and_logger(dummy_model_and_data, tmp_path):
    model, train_path, features = dummy_model_and_data
    db_path = tmp_path / "predictions.db"
    
    logger = PredictionLogger(db_path=str(db_path))
    sample_input = {feat: 0.5 for feat in features}
    logger.log_prediction(sample_input, 6.0)
    
    logged_df = logger.get_logged_predictions()
    assert len(logged_df) == 1
    assert "prediction" in logged_df.columns
    
    detector = DriftDetector(reference_data_path=train_path, db_path=str(db_path))
    report = detector.detect_drift(min_predictions=1)
    assert report["status"] == "success"
    assert "drift_detected" in report

def test_experiment_tracker():
    res = get_mlflow_runs()
    assert "enabled" in res
    assert "runs" in res

def test_analytics_and_pdf(dummy_model_and_data, tmp_path):
    model, train_path, features = dummy_model_and_data
    db_path = tmp_path / "predictions.db"
    
    logger = PredictionLogger(db_path=str(db_path))
    for i in range(5):
        sample_input = {feat: float(i)*0.1 for feat in features}
        logger.log_prediction(sample_input, 5.5 + i*0.1)
        
    import mlProject.components.analytics as analytics
    original_db = analytics.PredictionLogger
    
    try:
        analytics.PredictionLogger = lambda: logger
        
        summary = get_analytics_summary()
        assert summary["prediction_count"] == 5
        assert summary["mean_prediction"] > 0
        
        pdf_buffer = generate_pdf_report()
        assert pdf_buffer is not None
        assert len(pdf_buffer.getvalue()) > 0
    finally:
        analytics.PredictionLogger = original_db
