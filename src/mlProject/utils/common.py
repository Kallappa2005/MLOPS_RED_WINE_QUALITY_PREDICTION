import os
import hashlib
import zipfile
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


def load_env_file(env_path: Path = None):
    """Load .env file from the given path or default .env in project root."""
    dotenv_path = env_path or Path(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f"Loaded environment from {dotenv_path}")
    else:
        logger.info("No .env file found, using system environment variables")


def get_env_or_config(env_var: str, config_value, transform=None):
    """Return env var value if set, otherwise fall back to config value.
    
    Priority: Environment variable > .env > YAML config value.
    """
    env_val = os.environ.get(env_var)
    if env_val is not None:
        logger.info(f"Overriding config key {env_var} from environment")
        if transform:
            try:
                return transform(env_val)
            except (ValueError, TypeError):
                logger.warning(f"Could not transform env var {env_var}={env_val}, using config default")
                return config_value
        return env_val
    return config_value



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file is empty: {path_to_yaml}")
    except FileNotFoundError:
        logger.error(f"yaml file not found: {path_to_yaml}")
        raise
    except Exception as e:
        logger.exception(f"failed to read yaml file: {path_to_yaml}")
        raise
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            if verbose:
                logger.info(f"created directory at: {path}")
        except OSError as e:
            logger.error(f"failed to create directory at: {path} - {e}")
            raise


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def compute_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        path (Path): path to the file

    Returns:
        str: hex digest of SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@ensure_annotations
def save_checksum(model_path: Path, checksum_path: Path):
    """Compute SHA-256 of model file and save it to a sidecar file.

    Args:
        model_path (Path): path to the model file
        checksum_path (Path): path to the checksum sidecar file
    """
    checksum = compute_checksum(model_path)
    checksum_path.write_text(checksum)
    logger.info(f"Checksum saved to {checksum_path}")


@ensure_annotations
def verify_model_integrity(model_path: Path, checksum_path: Path) -> bool:
    """Verify model file integrity against its stored SHA-256 checksum.

    Args:
        model_path (Path): path to the model file
        checksum_path (Path): path to the checksum sidecar file

    Returns:
        bool: True if checksum matches, False otherwise
    """
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    if not checksum_path.exists():
        logger.error(
            f"Checksum file missing at {checksum_path} — "
            f"cannot verify model integrity"
        )
        return False
    expected = checksum_path.read_text().strip()
    actual = compute_checksum(model_path)
    if expected != actual:
        logger.error(
            f"Model integrity check FAILED for {model_path}. "
            f"Expected checksum: {expected}, Actual: {actual}"
        )
        return False
    logger.info(f"Model integrity verified for {model_path}")
    return True


@ensure_annotations
def safe_extract_zip(zip_path: Path, extract_dir: Path):
    """Extract a zip file safely, preventing Zip Slip path traversal.

    Validates each member's path to ensure it does not escape the
    target extraction directory via '..' or absolute paths.

    Args:
        zip_path (Path): path to the zip file
        extract_dir (Path): directory to extract into

    Raises:
        ValueError: if any member path is malicious (path traversal)
        zipfile.BadZipFile: if the zip file is corrupt
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"Zip Slip detected: member '{member}' would write "
                    f"outside the target directory"
                )
        zf.extractall(extract_dir)
    logger.info(f"Safely extracted zip to {extract_dir}")


@ensure_annotations
def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify a file's SHA-256 checksum against an expected value.

    Args:
        file_path (Path): path to the file
        expected_checksum (str): expected SHA-256 hex digest

    Returns:
        bool: True if checksum matches (or expected is empty), False otherwise
    """
    if not expected_checksum:
        logger.info("No expected checksum configured - skipping verification")
        return True
    if not file_path.exists():
        logger.error(f"File not found for checksum verification: {file_path}")
        return False
    actual = compute_checksum(file_path)
    if actual != expected_checksum:
        logger.error(
            f"Checksum mismatch for {file_path}: "
            f"expected {expected_checksum}, got {actual}"
        )
        return False
    logger.info(f"Checksum verified for {file_path}")
    return True
