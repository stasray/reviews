import sys
import types
from pathlib import Path
import importlib

import numpy as np
import pytest


@pytest.fixture(scope="session")
def services_mod():
    try:
        import sklearn  # noqa: F401
    except Exception:
        pytest.skip("scikit-learn not installed; skipping services-dependent tests")

    if "sentence_transformers" not in sys.modules:
        m = types.ModuleType("sentence_transformers")

        class _DummyST:
            def __init__(self, name: str):
                self.name = name

            def encode(self, texts, normalize_embeddings: bool = False):
                n = len(texts)
                # Return deterministic zeros; shape must be 2D
                return np.zeros((n, 8), dtype=float)

        m.SentenceTransformer = _DummyST
        sys.modules["sentence_transformers"] = m

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.modules.setdefault("models", importlib.import_module("src.models"))
    sys.modules.setdefault("insights", importlib.import_module("src.insights"))

    import src.services as _services
    return _services
