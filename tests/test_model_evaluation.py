import numpy as np

from mlProject.components.model_evaluation import ModelEvaluation


def test_baseline_r2_is_zero_for_integer_targets():
    """The predict-the-mean baseline must score R2=0 even for integer targets.

    quality is int64 in schema.yaml, so the baseline must not let the mean be
    truncated to an integer (which would push R2 below zero).
    """
    evaluator = ModelEvaluation(config=None)
    actual = np.array([5, 6, 7, 5, 6, 8, 4, 5, 6, 7])

    baseline_r2 = evaluator.baseline_r2_score(actual)

    assert abs(baseline_r2) < 1e-9
