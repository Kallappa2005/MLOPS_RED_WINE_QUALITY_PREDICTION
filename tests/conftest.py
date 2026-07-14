import os
import sys

# Add the project root to sys.path so that 'app' module can be imported by tests
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set predictable environment variables for tests before app/security is imported
os.environ["ADMIN_PASSWORD"] = "admin_password"
os.environ["ENGINEER_PASSWORD"] = "engineer_password"
os.environ["VIEWER_PASSWORD"] = "viewer_password"
os.environ["JWT_SECRET"] = "test_secret_key"
