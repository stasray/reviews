import io
from types import SimpleNamespace

import pandas as pd
import pytest


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
    # Contains a meaningful concept or prefix
    assert label_contains.lower() in name.lower()


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


def test_hf_sentiment_batch_parsing_and_logs(monkeypatch, services_mod):
    # Simulate HF response: list[list[dict]] per input
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
    assert "generated_text" not in s  # returns the string, not raw JSON
    # Log captured
    assert services_mod.get_hf_logs(1)


def test_analyze_sentiment_ai_auto_ru_flip(monkeypatch, services_mod):
    # Force RU route and positive result; heuristic should flip to negative due to strong negative words
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
    # TSV
    df = pd.DataFrame({"text": ["a", "b"]})
    buf = io.BytesIO()
    df.to_csv(buf, sep="\t", index=False)
    reviews = services_mod.load_reviews_from_file(buf.getvalue(), "file.tsv")
    assert [r.text for r in reviews] == ["a", "b"]

    # XLSX
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
    # Deduped and normalized, drop junk "item"
    low = [x.lower() for x in out]
    assert any("late delivery" in x for x in low)
    assert any(("плохое качество" in x) or ("poor quality" in x) for x in low)
    assert any("поврежденная упаковка" in x for x in low)
