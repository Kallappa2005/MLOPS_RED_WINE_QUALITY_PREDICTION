import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from mlProject import logger
from mlProject.entity.config_entity import DataDriftConfig

class DataDrift:
    def __init__(self, config: DataDriftConfig):
        self.config = config

    def detect_drift(self):
        try:
            logger.info("Starting Data Drift detection")
            
            if not self.config.reference_data_path.exists():
                logger.warning(f"Reference data not found at {self.config.reference_data_path}. Skipping data drift check.")
                return

            reference_data = pd.read_csv(self.config.reference_data_path)
            current_data = pd.read_csv(self.config.current_data_path)
            
            # Initialize Evidently Report
            data_drift_report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            data_drift_report.run(reference_data=reference_data, current_data=current_data)
            
            # Save HTML report
            data_drift_report.save_html(str(self.config.report_path))
            logger.info(f"Data Drift report saved at: {self.config.report_path}")
            
            # Extract dictionary to check for drift share
            report_dict = data_drift_report.as_dict()
            dataset_drift = report_dict['metrics'][0]['result']['dataset_drift']
            share_of_drifted_columns = report_dict['metrics'][0]['result']['share_of_drifted_columns']
            
            logger.info(f"Drift share: {share_of_drifted_columns}, threshold: {self.config.drift_share_threshold}")
            
            if share_of_drifted_columns > self.config.drift_share_threshold:
                error_msg = f"Data Drift Detected! Share of drifted columns ({share_of_drifted_columns}) exceeds threshold ({self.config.drift_share_threshold})."
                logger.error(error_msg)
                raise Exception(error_msg)
            else:
                logger.info("No significant Data Drift detected. Pipeline can proceed.")
                
        except Exception as e:
            logger.exception(e)
            raise e
