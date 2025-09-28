import io
import pandas as pd
import pytest

from types import SimpleNamespace


def test_detect_language_basic(services_mod):
    assert services_mod.detect_language("Привет, мир") == "ru"
    assert services_mod.detect_language("Hello, world") == "en"
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
    df = pd.DataFrame({"Отзыв": ["Отличное качество", "Доставка опоздала"]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    # Re-encode into cp1251 bytes to emulate uploaded file
    data = buf.getvalue().encode("cp1251", errors="ignore")
    reviews = services_mod.load_reviews_from_file(data, "reviews.csv")
    texts = [r.text for r in reviews]
    assert "Отличное качество" in texts and "Доставка опоздала" in texts


def test_load_reviews_from_file_unsupported_format_raises(services_mod):
    data = b"some text content"
    with pytest.raises(ValueError):
        services_mod.load_reviews_from_file(data, "notes.txt")


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
    monkeypatch.setattr(services_mod, "extract_key_insights", lambda reviews, max_items=3: {"problems": ["Late delivery"], "strengths": ["Great quality"]})
    monkeypatch.setattr(services_mod, "clarify_insights", lambda reviews, insights, token=None: insights)

    reviews = [
        SimpleNamespace(text="Delivery was late", sentiment="", sentiment_score=0.0),
        SimpleNamespace(text="Excellent quality", sentiment="", sentiment_score=0.0),
        SimpleNamespace(text="Цена нормальная", sentiment="", sentiment_score=0.0),
    ]

    analysis = services_mod.run_analysis(reviews, use_ai=False)
    total = sum(analysis.stats["sentiment_counts"].values())
    assert total == len(reviews)
    for r in analysis.reviews:
        assert r.sentiment in {"positive", "negative", "neutral"}
        assert isinstance(r.topic_id, int)
        assert isinstance(r.topic, str)
    assert analysis.stats["insights"]["problems"]
    assert analysis.stats["insights"]["strengths"]


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
    monkeypatch.setattr(
        services_mod,
        "_hf_sentiment_batch",
        lambda texts, model, token: [("positive", 0.6) for _ in texts],
    )
    reviews = [SimpleNamespace(text="Good product", language="en", sentiment="", sentiment_score=0.0)]
    services_mod.analyze_sentiment_ai(reviews, model="some/model", token="t", batch_size=8)
    assert reviews[0].sentiment == "positive"


def test_run_analysis_ai_label_normalization_counts(monkeypatch, services_mod):
    labels = ["label_1", "label_3", "label_4", "5 stars", "1 star"]
    def fake_batch(texts, model, token):
        out = []
        for i in range(len(texts)):
            lab = labels[i % len(labels)]
            out.append((lab, 0.5))
        return out
    monkeypatch.setattr(services_mod, "_hf_sentiment_batch", fake_batch)
    monkeypatch.setattr(services_mod, "extract_key_insights", lambda *a, **k: {"problems": ["Late delivery"], "strengths": ["Great quality"]})
    monkeypatch.setattr(services_mod, "clarify_insights", lambda *a, **k: {"problems": ["Late delivery"], "strengths": ["Great quality"]})

    reviews = [SimpleNamespace(text=f"t{i}", language="en", sentiment="", sentiment_score=0.0) for i in range(10)]
    analysis = services_mod.run_analysis(reviews, use_ai=True, hf_model="x", hf_token="t")
    counts = analysis.stats["sentiment_counts"]
    assert set(counts.keys()) == {"positive", "neutral", "negative"}
    assert counts["positive"] > 0 and counts["negative"] > 0


def test_load_reviews_from_file_fallback_first_column(services_mod):
    df = pd.DataFrame({"other": ["hello", "world"]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    reviews = services_mod.load_reviews_from_file(buf.getvalue(), "file.csv")
    assert [r.text for r in reviews] == ["hello", "world"]


def test_normalize_phrase_en_ru(services_mod):
    n = services_mod._normalize_phrase
    assert "late delivery" in n("arrived late, delivery was delayed").lower()
    assert n("damaged package box").startswith("Damaged packaging")
    assert "поврежденная упаковка" in n("порвана упаковка и повреждена коробка").lower()
    assert "опоздание доставки" in n("опоздала доставка").lower()


@pytest.mark.parametrize(
    "keywords,sent,label_contains",
    [
        (["delivery", "late"], "negative", "Poor"),
        (["quality", "excellent"], "positive", "Good"),
        (["цена", "дорого"], "neutral", "Цена"),
    ],
)
def test_create_topic_name(services_mod, keywords, sent, label_contains):
    name = services_mod._create_topic_name(keywords, sentiment=sent)
    assert isinstance(name, str) and name
    assert label_contains.lower() in name.lower()


def test_hf_sentiment_batch_parsing_and_logs(monkeypatch, services_mod):
    data = [
        [
            {"label": "5 stars", "score": 0.9},
            {"label": "1 star", "score": 0.1},
        ],
        [
            {"label": "label_2", "score": 0.8},
            {"label": "label_4", "score": 0.2},
        ],
    ]

    monkeypatch.setattr(services_mod.requests, "post", lambda *a, **k: _fake_resp(data))
    out = services_mod._hf_sentiment_batch(["a", "b"], model="m", token="t")
    assert out == [("positive", 0.9), ("negative", 0.8)]
    logs = services_mod.get_hf_logs(5)
    assert isinstance(logs, list) and len(logs) >= 1


def test_hf_text2text_generated_text_parsing(monkeypatch, services_mod):
    data = [{"generated_text": '{"problems": ["item broken"], "strengths": ["great quality"]}'}]
    monkeypatch.setattr(services_mod.requests, "post", lambda *a, **k: _fake_resp(data))
    s = services_mod._hf_text2text("prompt", model="m", token="t")
    assert "generated_text" not in s
    assert services_mod.get_hf_logs(1)


def test_analyze_sentiment_ai_auto_ru_flip(monkeypatch, services_mod):
    monkeypatch.setattr(
        services_mod,
        "_hf_sentiment_batch",
        lambda texts, model, token: [("positive", 0.7) for _ in texts],
    )
    reviews = [
        SimpleNamespace(text="Ужасный товар, полный кошмар!!!", language="ru", sentiment="", sentiment_score=0.0),
    ]
    services_mod.analyze_sentiment_ai(reviews, model="auto (recommended)", token="t", batch_size=8)
    assert reviews[0].sentiment == "negative"
    assert reviews[0].sentiment_score >= 0.6


def test_load_reviews_from_file_tsv_and_xlsx(services_mod):
    df = pd.DataFrame({"text": ["a", "b"]})
    buf = io.BytesIO()
    df.to_csv(buf, sep="\t", index=False)
    reviews = services_mod.load_reviews_from_file(buf.getvalue(), "file.tsv")
    assert [r.text for r in reviews] == ["a", "b"]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        pd.DataFrame({"review": ["x", "y"]}).to_excel(writer, index=False)
    reviews = services_mod.load_reviews_from_file(buf.getvalue(), "file.xlsx")
    assert [r.text for r in reviews] == ["x", "y"]


def test_clean_insight_list_dedup_and_normalize(services_mod):
    items = [
        "- arrived late",
        "late delivery",
        "item",
        "poor quality",
        "повреждена упаковка",
    ]
    out = services_mod._clean_insight_list(items, polarity="negative")
    low = [x.lower() for x in out]
    assert any("late delivery" in x for x in low)
    assert any(("плохое качество" in x) or ("poor quality" in x) for x in low)
    assert any("поврежденная упаковка" in x for x in low)
