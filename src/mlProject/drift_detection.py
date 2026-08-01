import pandas as pd
from scipy.stats import ks_2samp
from mlProject import logger
def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.05):
    """Detects data drift using Kolmogorov-Smirnov test"""
    drift_detected = {}
    for column in reference_data.columns:
        if column in current_data.columns:
            stat, p_value = ks_2samp(reference_data[column], current_data[column])
            drift_detected[column] = p_value < threshold
            
    if any(drift_detected.values()):
        logger.warning("Data drift detected in features!")
        
    return drift_detected