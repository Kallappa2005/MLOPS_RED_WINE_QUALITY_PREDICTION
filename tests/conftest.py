import pytest
import sys
from pathlib import Path
import hashlib
from werkzeug.security import generate_password_hash

# Patch hashlib.md5 and hashlib.new to avoid TypeError on Python 3.8 / older environment
original_md5 = hashlib.md5
def patched_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return original_md5(*args, **kwargs)
hashlib.md5 = patched_md5

original_new = hashlib.new
def patched_new(name, *args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return original_new(name, *args, **kwargs)
hashlib.new = patched_new

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def mock_user_db(monkeypatch):
    """Patch USER_DB in all modules that use it for tests."""
    test_user_db = {
        "admin": {
            "password": "admin_password",
            "password_hash": generate_password_hash("admin_password"),
            "role": "Admin"
        },
        "engineer": {
            "password": "engineer_password",
            "password_hash": generate_password_hash("engineer_password"),
            "role": "Engineer"
        },
        "viewer": {
            "password": "viewer_password",
            "password_hash": generate_password_hash("viewer_password"),
            "role": "Viewer"
        }
    }
    
    # Patch in security module
    try:
        from mlProject.components import security
        monkeypatch.setattr(security, "USER_DB", test_user_db)
    except ImportError:
        pass
    
    # Patch in app module
    try:
        import app
        monkeypatch.setattr(app, "USER_DB", test_user_db)
    except ImportError:
        pass

