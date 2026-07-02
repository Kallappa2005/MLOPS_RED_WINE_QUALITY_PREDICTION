import os
import urllib.request as request
from mlProject import logger
from mlProject.utils.common import get_size, safe_extract_zip, verify_checksum
from mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_data_file
                )
                logger.info(f"{filename} download! with following info: \n{headers}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

            if not verify_checksum(
                Path(self.config.local_data_file),
                self.config.expected_checksum
            ):
                raise ValueError(
                    f"Checksum verification failed for {self.config.local_data_file}"
                )
        except Exception as e:
            logger.exception(f"Failed to download file from {self.config.source_URL}")
            raise



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            safe_extract_zip(Path(self.config.local_data_file), Path(unzip_path))
        except ValueError as e:
            logger.error(f"Zip Slip detected in {self.config.local_data_file}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Failed to extract zip file: {self.config.local_data_file}")
            raise
# Added missing docstrings
