import io
import json
import types

import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace


def test_detect_language_basic(services_mod):
    assert services_mod.detect_language("Привет, мир") == "ru"
    assert services_mod.detect_language("Hello, world") == "en"
    # Mixed but with Cyrillic -> ru
    assert services_mod.detect_language("Hello мир") == "ru"


@pytest.mark.parametrize(
    "text,lang,expected_label",
    [
        ("excellent quality, very satisfied", "en", "positive"),
        ("delivery was late and damaged box", "en", "negative"),
        ("Цена нормальная, все ок", "ru", "neutral"),
        ("Доставка опоздала и упаковка порвана", "ru", "negative"),
        ("Отличное качество, рекомендую", "ru", "positive"),
    ],
)
def test_simple_sentiment_score(services_mod, text, lang, expected_label):
    label, score = services_mod.simple_sentiment_score(text, lang)
    assert label == expected_label
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("positive", "positive"),
        ("POS", "positive"),
        ("neg", "negative"),
        ("neutral", "neutral"),
        ("5 stars", "positive"),
        ("1 star", "negative"),
        ("label_4", "positive"),
        ("LABEL-2", "negative"),
        ("something else", "neutral"),
    ],
)
def test_normalize_hf_label(services_mod, raw, expected):
    assert services_mod._normalize_hf_label(raw) == expected


def test_load_reviews_from_file_csv_utf8(services_mod):
    df = pd.DataFrame({"text": ["hello", "world", " "]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue()
    reviews = services_mod.load_reviews_from_file(data, "sample.csv")
    assert [r.text for r in reviews] == ["hello", "world"]


def test_load_reviews_from_file_csv_cp1251_with_russian_header(services_mod):
    # Column named in Russian ("Отзыв") and Windows-1251 encoding
    df = pd.DataFrame({"Отзыв": ["Отличное качество", "Доставка опоздала"]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    # Re-encode into cp1251 bytes to emulate uploaded file
    data = buf.getvalue().encode("cp1251", errors="ignore")
    reviews = services_mod.load_reviews_from_file(data, "reviews.csv")
    texts = [r.text for r in reviews]
    assert "Отличное качество" in texts and "Доставка опоздала" in texts


def test_extract_topics_basic(services_mod):
    texts = [
        "Delivery was late and the courier was rude",
        "Excellent product quality, very satisfied",
        "Package was damaged, box was open",
        "Support helped quickly with my issue",
    ]
    reviews = [SimpleNamespace(text=t, sentiment=("negative" if i % 2 == 0 else "positive")) for i, t in enumerate(texts)]
    topic_ids, topics_info = services_mod.extract_topics(texts, reviews=reviews, n_topics=3, n_keywords=4)
    assert len(topic_ids) == len(texts)
    assert isinstance(topics_info, dict)
    # At least one topic must have a non-empty label
    assert any(v.get("label") for v in topics_info.values())


def test_run_analysis_rule_based_monkeypatched_insights(services_mod, monkeypatch):
    # Avoid heavy SentenceTransformer usage inside insights by patching services' imported symbols
    monkeypatch.setattr(services_mod, "extract_key_insights", lambda reviews, max_items=3: {"problems": ["Late delivery"], "strengths": ["Great quality"]})
    monkeypatch.setattr(services_mod, "clarify_insights", lambda reviews, insights, token=None: insights)

    reviews = [
        SimpleNamespace(text="Delivery was late", sentiment="", sentiment_score=0.0),
        SimpleNamespace(text="Excellent quality", sentiment="", sentiment_score=0.0),
        SimpleNamespace(text="Цена нормальная", sentiment="", sentiment_score=0.0),
    ]

    analysis = services_mod.run_analysis(reviews, use_ai=False)
    # Sentiment counts sum to total
    total = sum(analysis.stats["sentiment_counts"].values())
    assert total == len(reviews)
    # All reviews have normalized sentiment and topic assigned
    for r in analysis.reviews:
        assert r.sentiment in {"positive", "negative", "neutral"}
        assert isinstance(r.topic_id, int)
        assert isinstance(r.topic, str)
    # Insights injected by monkeypatch present
    assert analysis.stats["insights"]["problems"]
    assert analysis.stats["insights"]["strengths"]
