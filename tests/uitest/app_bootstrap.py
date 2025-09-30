import sys
from pathlib import Path
import importlib

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

sys.modules.setdefault("models", importlib.import_module("src.models"))
sys.modules.setdefault("insights", importlib.import_module("src.insights"))
sys.modules.setdefault("services", importlib.import_module("src.services"))

import src.app  # noqa: F401

