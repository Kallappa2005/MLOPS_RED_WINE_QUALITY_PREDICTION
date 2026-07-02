import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def mock_user_db(monkeypatch):
    """Patch USER_DB in all modules that use it for tests."""
    # Test user DB with 'password' key (plain text for testing)
    test_user_db = {
        "admin": {"password": "admin_password", "role": "Admin"},
        "engineer": {"password": "engineer_password", "role": "Engineer"},
        "viewer": {"password": "viewer_password", "role": "Viewer"}
    }
    
    # Patch in security module
    from mlProject.components import security
    monkeypatch.setattr(security, "USER_DB", test_user_db)
    
    # Patch in app module (which imports USER_DB at import time)
    import app
    monkeypatch.setattr(app, "USER_DB", test_user_db)
