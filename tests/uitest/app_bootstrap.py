import sys
from pathlib import Path
import importlib

# Ensure project root on sys.path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Map absolute imports used by the app to package modules
sys.modules.setdefault("models", importlib.import_module("src.models"))
sys.modules.setdefault("insights", importlib.import_module("src.insights"))
sys.modules.setdefault("services", importlib.import_module("src.services"))

# Import the real Streamlit app (executes at import time)
import src.app  # noqa: F401

