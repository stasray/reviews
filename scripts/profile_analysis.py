import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path


def _ensure_import_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    try:
        sys.modules.setdefault("models", importlib.import_module("src.models"))
    except Exception:
        pass
    try:
        sys.modules.setdefault("insights", importlib.import_module("src.insights"))
    except Exception:
        pass


def _stub_sentence_transformers():
    # Avoid heavy model downloads and speed up profiling by stubbing
    import types
    import numpy as np
    if "sentence_transformers" not in sys.modules:
        m = types.ModuleType("sentence_transformers")

        class _DummyST:
            def __init__(self, name: str):
                self.name = name

            def encode(self, texts, normalize_embeddings: bool = False):
                n = len(texts)
                return np.zeros((n, 8), dtype=float)

        m.SentenceTransformer = _DummyST
        sys.modules["sentence_transformers"] = m


def _lightweight_insights_patch(svc_mod):
    # Replace heavy insights functions with lightweight stubs
    svc_mod.extract_key_insights = lambda reviews, max_items=3: {
        "problems": ["Late delivery", "Damaged packaging"][:max_items],
        "strengths": ["Great quality", "Fast support"][:max_items],
    }
    svc_mod.clarify_insights = lambda reviews, insights, token=None: insights


def profile_run(sizes):
    _ensure_import_path()
    _stub_sentence_transformers()

    from src import services as svc

    _lightweight_insights_patch(svc)

    results = []
    for n in sizes:
        reviews = svc.generate_fake_reviews(n)
        tracemalloc.start()
        t0 = time.perf_counter()
        analysis = svc.run_analysis(reviews, use_ai=False)
        dt = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({
            "n_reviews": n,
            "time_sec": round(dt, 4),
            "reviews_per_sec": round(n / dt if dt > 0 else float("inf"), 2),
            "peak_mem_mb": round(peak / (1024 * 1024), 2),
            "topics": len(analysis.stats.get("topics", {})),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile run_analysis performance")
    parser.add_argument(
        "--sizes",
        default="50,200,500",
        help="Comma-separated review counts to test (e.g., 50,200,500)",
    )
    parser.add_argument(
        "--out",
        default="profile_results.json",
        help="Path to write JSON results",
    )
    args = parser.parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    results = profile_run(sizes)
    print("Load test results:")
    for r in results:
        print(
            f"n={r['n_reviews']:4d}  time={r['time_sec']:7.3f}s  rps={r['reviews_per_sec']:8.2f}  peak_mem={r['peak_mem_mb']:6.2f}MB  topics={r['topics']}"
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {args.out}")


if __name__ == "__main__":
    main()
