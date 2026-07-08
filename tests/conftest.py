import sys
import os

# Add the src directory to the Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add the project root to the Python path to allow absolute imports like 'from app import app'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
