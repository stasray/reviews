import io
from types import SimpleNamespace

import pandas as pd


def _fake_resp(json_obj, status=200):
    class R:
        status_code = status

        def json(self):
            return json_obj

        @property
        def text(self):
            return ""

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError("http error")

        @property
        def headers(self):
            return {"content-length": "0"}

        @property
        def content(self):
            return b""

        @property
        def ok(self):
            return status < 400

    return R()


def test_ru_negative_heuristic(services_mod):
    f = services_mod._ru_negative_heuristic
    assert f("Это просто ужас и кошмар!!") is True
    assert f("Очень доволен покупкой") is False


def test_hf_sentiment_batch_single_result_fix(monkeypatch, services_mod):
    # API returns a single item list for multiple texts
    data = [[{"label": "1 star", "score": 0.7}, {"label": "5 stars", "score": 0.8}]]
    monkeypatch.setattr(services_mod.requests, "post", lambda *a, **k: _fake_resp(data))
    out = services_mod._hf_sentiment_batch(["a", "b"], model="m", token="t")
    assert out == [("negative", 0.7), ("positive", 0.8)]


def test_hf_text2text_dict_payload(monkeypatch, services_mod):
    data = {"generated_text": "result text"}
    monkeypatch.setattr(services_mod.requests, "post", lambda *a, **k: _fake_resp(data))
    s = services_mod._hf_text2text("p", model="m", token="t")
    assert s == "result text"


def test_analyze_sentiment_ai_single_model_path(monkeypatch, services_mod):
    # Force single-model branch, not auto routing
    monkeypatch.setattr(
        services_mod,
        "_hf_sentiment_batch",
        lambda texts, model, token: [("positive", 0.6) for _ in texts],
    )
    reviews = [SimpleNamespace(text="Good product", language="en", sentiment="", sentiment_score=0.0)]
    services_mod.analyze_sentiment_ai(reviews, model="some/model", token="t", batch_size=8)
    assert reviews[0].sentiment == "positive"


def test_run_analysis_ai_label_normalization_counts(monkeypatch, services_mod):
    # Use AI path with normalized labels
    labels = ["label_1", "label_3", "label_4", "5 stars", "1 star"]
    def fake_batch(texts, model, token):
        out = []
        for i in range(len(texts)):
            lab = labels[i % len(labels)]
            out.append((lab, 0.5))
        return out
    monkeypatch.setattr(services_mod, "_hf_sentiment_batch", fake_batch)
    # Also avoid heavy insights
    monkeypatch.setattr(services_mod, "extract_key_insights", lambda *a, **k: {"problems": ["Late delivery"], "strengths": ["Great quality"]})
    monkeypatch.setattr(services_mod, "clarify_insights", lambda *a, **k: {"problems": ["Late delivery"], "strengths": ["Great quality"]})

    reviews = [SimpleNamespace(text=f"t{i}", language="en", sentiment="", sentiment_score=0.0) for i in range(10)]
    analysis = services_mod.run_analysis(reviews, use_ai=True, hf_model="x", hf_token="t")
    counts = analysis.stats["sentiment_counts"]
    assert set(counts.keys()) == {"positive", "neutral", "negative"}
    # We expect a mix (at least one positive and one negative)
    assert counts["positive"] > 0 and counts["negative"] > 0


def test_load_reviews_from_file_fallback_first_column(services_mod):
    df = pd.DataFrame({"other": ["hello", "world"]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    reviews = services_mod.load_reviews_from_file(buf.getvalue(), "file.csv")
    assert [r.text for r in reviews] == ["hello", "world"]
