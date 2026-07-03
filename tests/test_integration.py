import pytest
from mlProject.pipeline.prediction import PredictionPipeline
import pandas as pd

def test_end_to_end_prediction():
    # Dummy integration test
    pipeline = PredictionPipeline()
    # Ensure pipeline is initialized successfully
    assert pipeline is not None
