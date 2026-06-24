import pytest
import os
from mlProject.utils.common import read_yaml, create_directories
from pathlib import Path

def test_create_directories(tmp_path):
    # Test directory creation
    test_dir = tmp_path / "test_dir"
    create_directories([test_dir])
    assert os.path.exists(test_dir)

def test_read_yaml_failure():
    # Test reading non-existent yaml
    with pytest.raises(Exception):
        read_yaml(Path("non_existent_file.yaml"))
