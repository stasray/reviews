import random
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import re
import json
from sentence_transformers import SentenceTransformer

# NOTE: Heuristic/AI/Embedding insights moved to insights.py to keep this file smaller.
from insights import (
    extract_insights_heuristic,
    extract_insights_ai,
    extract_key_insights,
    _clean_insight_list,
    clarify_insights,
)

from models import Review, Analysis

# Simple in-memory log for HF API responses (latest first)
_HF_API_LOGS: list = []


def get_hf_logs(limit: int = 10):
    return _HF_API_LOGS[:limit]


def _hf_text2text(prompt: str, model: str, token: str, max_new_tokens: int = 512) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2}
    }
    # Retry with exponential backoff
    backoffs = [1, 2, 4]
    last_exc = None
    for attempt, delay in enumerate([0] + backoffs):
        if delay:
            try:
                import time as _time
                _time.sleep(delay)
            except Exception:
                pass
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            status = response.status_code
            try:
                data = response.json()
            except Exception:
                data = {"non_json": response.text}
            response.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            log_entry = {
                "url": url,
                "status": locals().get("status", None),
                "inputs_count": 1,
                "error": str(e),
                "response": locals().get("data", None),
            }
            _HF_API_LOGS.insert(0, log_entry)
            if len(_HF_API_LOGS) > 50:
                del _HF_API_LOGS[50:]
            print("[HF DEBUG ERROR]", log_entry)
            if attempt == len(backoffs):
                raise
    log_entry = {
        "url": url,
        "status": status,
        "inputs_count": 1,
        "response": data,
    }
    _HF_API_LOGS.insert(0, log_entry)
    if len(_HF_API_LOGS) > 50:
        del _HF_API_LOGS[50:]
    print("[HF DEBUG]", log_entry)
    # data usually: [{"generated_text": "..."}] or {"generated_text": "..."}
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"]) or ""
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]) or ""
    # fallback
    return json.dumps(data)


def _parse_insights_from_text(text: str) -> Dict[str, List[str]]:
    problems: List[str] = []
    strengths: List[str] = []
    # Try JSON first
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            obj = json.loads(m.group(0))
            p = obj.get("problems") or obj.get("issues") or []
            s = obj.get("strengths") or obj.get("pros") or []
            problems = [str(x).strip() for x in p if str(x).strip()]
            strengths = [str(x).strip() for x in s if str(x).strip()]
    except Exception:
        problems = []
        strengths = []
    # If not parsed, try bullets
    if not problems and not strengths:
        # split by lines, detect sections
        lines = [l.strip(" -*•\t") for l in text.splitlines()]
        mode = None
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["problems", "issues", "недостат", "минус"]):
                mode = "problems"
                continue
            if any(k in low for k in ["strengths", "pros", "достоин", "плюс"]):
                mode = "strengths"
                continue
            if mode == "problems" and len(line) >= 3:
                problems.append(line)
            elif mode == "strengths" and len(line) >= 3:
                strengths.append(line)
    return {"problems": problems[:10], "strengths": strengths[:10]}


def extract_insights_ai(reviews: List[Review], token: str, model: Optional[str] = None) -> Dict[str, List[str]]:
    if not token:
        return {"problems": [], "strengths": []}
    model_name = model or "google/flan-t5-large"
    # Prepare multilingual prompt
    # Keep prompt reasonably small
    snippets = []
    total_chars = 0
    for r in reviews:
        t = (r.text or "").strip()
        if not t:
            continue
        t = t.replace("\n", " ")
        if len(t) > 400:
            t = t[:400] + "…"
        if total_chars + len(t) > 6000:
            break
        snippets.append(f"- {t}")
        total_chars += len(t)
    has_cyr = any("\u0400" <= ch <= "\u04FF" for ch in " ".join(snippets))
    if has_cyr:
        system = (
            "Извлеки ключевые ПРОБЛЕМЫ и ДОСТОИНСТВА из отзывов ниже. "
            "Сгруппируй по смыслу, избегай повторов, используй краткие пунктов списки. "
            "Ответ верни в JSON с полями problems (список строк) и strengths (список строк)."
        )
        prompt = system + "\nОтзывы:\n" + "\n".join(snippets)
    else:
        system = (
            "Extract the key PROBLEMS and STRENGTHS from the reviews below. "
            "Group similar points, avoid duplicates, use concise bullet-like phrases. "
            "Return JSON with fields problems (list of strings) and strengths (list of strings)."
        )
        prompt = system + "\nReviews:\n" + "\n".join(snippets)
    try:
        gen = _hf_text2text(prompt, model_name, token, max_new_tokens=384)
        parsed = _parse_insights_from_text(gen)
        # Post-process AI output to improve quality
        cleaned = {
            "problems": _clean_insight_list(parsed.get("problems", []), polarity="negative"),
            "strengths": _clean_insight_list(parsed.get("strengths", []), polarity="positive"),
        }
        return cleaned
    except Exception:
        return {"problems": [], "strengths": []}


def extract_insights_heuristic(reviews: List[Review]) -> Dict[str, List[str]]:
    # Split corpora by sentiment
    pos_texts = [r.text for r in reviews if (r.sentiment or "").startswith("pos")]
    neg_texts = [r.text for r in reviews if (r.sentiment or "").startswith("neg")]
    return {pos_texts: neg_texts}
    # Basic stopwords (en+ru)

stop = {"and", "the", "to", "of", "a", "in", "for", "is", "on", "it", "was", "with", "this", "that", "are", "be",
        "as", "have", "but", "not", "or", "very", "i", "we", "you", "they", "them", "their", "our", "my", "your",
        "me", "he", "she", "his", "her", "its", "at", "by", "from", "so", "if", "because", "also", "even", "still",
        "all", "really", "just", "one", "get", "got", "would", "could", "should", "arrived", "item", "thing",
        "stuff", "lot", "bit", "quite", "pretty", "и", "в", "на", "это", "как", "что", "не", "с", "за", "от",
        "очень", "бы", "то", "по", "у", "из", "а", "но", "до", "при", "для", "же", "так", "он", "она", "они", "мы",
        "вы", "мне", "нам", "вас", "их", "там", "тут", "ещё", "еще", "очень", "всего", "все", "реально", "просто",
        "вот", "типа", "короче", "товар", "вещь", "предмет"}

def top_phrases(texts: List[str], n: int = 8) -> List[str]:
    if not texts:
        return []
    # Prefer multi-word keyphrases
    vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(2, 3),
        min_df=1,
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X = vec.fit_transform([(t or "").lower() for t in texts])
    vocab = vec.get_feature_names_out()
    scores = X.mean(axis=0).A1
    pairs = [(vocab[i], scores[i]) for i in range(len(vocab))]
    def ok(term: str) -> bool:
        parts = term.split()
        # Reject phrases that contain stopwords or single-letter tokens
        if any(len(w) <= 2 or w.isdigit() or w in stop for w in parts):
            return False
        # Reject phrases made of the same word repeated
        if len(set(parts)) == 1:
            return False
        # Prefer phrases that include at least one content-looking token
        if not any(re.search(r"[a-zA-Zа-яА-Я]", w) for w in parts):
            return False
        return True
    pairs = [(t, s) for (t, s) in pairs if ok(t)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    # Diversity-aware selection with simple MMR-like penalization by word overlap
    result: List[str] = []
    result_norm: List[str] = []
    seen_keys: set = set()
    for term, _ in pairs:
        norm = _normalize_phrase(term)
        if not norm:
            continue
        # Skip if looks too generic after normalization
        generic = {"Really", "All", "Item", "Товар", "Просто", "Очень"}
        if norm in generic:
            continue
        key = " ".join(sorted(set(norm.lower().split())))
        if key in seen_keys:
            continue
        # Penalize high overlap with already selected phrases
        words = set(norm.lower().split())
        too_similar = False
        for existing in result_norm:
            w2 = set(existing.lower().split())
            overlap = len(words & w2) / max(1, len(words | w2))
            if overlap >= 0.6:
                too_similar = True
                break
        if too_similar:
            continue
        seen_keys.add(key)
        result.append(norm)
        result_norm.append(norm)
        if len(result) >= n:
            break
    return result

def _normalize_phrase(phrase: str) -> str:
    t = (phrase or "").strip().lower()
    # English normalizations
    t = re.sub(r"\b(arriv(e|ed|es|ing))\s+(late|delayed|with delay)\b", "late delivery", t)
    t = re.sub(r"\b(delivery\s+(was\s+)?)?(late|delayed)\b", "late delivery", t)
    t = re.sub(r"\b(damag(e|ed|es|ing))\s+(package|packaging|box|item|product)\b", "damaged packaging", t)
    t = re.sub(r"\b(broken|cracked)\s+(item|product|screen|device)\b", "broken product", t)
    t = re.sub(r"\b(bad|poor)\s+quality\b", "poor quality", t)
    t = re.sub(r"\b(high|expensive)\s+price\b", "high price", t)
    t = re.sub(r"\b(rude|unhelpful)\s+(support|customer service|service|courier|driver)\b", "poor service", t)
    t = re.sub(r"\b(long|slow)\s+(response|reply|shipping|delivery)\b", "slow service", t)
    t = re.sub(r"\b(missing|lost)\s+(parts|item|package|order)\b", "missing items", t)
    # Russian normalizations
    t = re.sub(r"\b(доставк[аи])\s+(опоздал[аи]|поздн[ао])\b", "опоздание доставки", t)
    t = re.sub(r"\b(опоздал[аи])\s+доставк[аи]\b", "опоздание доставки", t)
    t = re.sub(r"\b(порв[ао]н(ая|ы)|помят(ая|ы)|поврежд(ен|ена|ены))\s+(упаковк[аи]|коробк[аи])\b", "поврежденная упаковка", t)
    t = re.sub(r"\b(сломан(а|о|ы)|трещин[аы])\b", "сломанный товар", t)
    t = re.sub(r"\b(плох(ое|ой)|низкое)\s+качеств[оа]\b", "плохое качество", t)
    t = re.sub(r"\b(высокая|завышенная|большая)\s+цен[аы]\b", "высокая цена", t)
    t = re.sub(r"\b(груб(ая|ый)|хамств(о|о)|невежлив(ая|ый))\s+(поддержк[аи]|курьер|сервис)\b", "плохой сервис", t)
    t = re.sub(r"\b(долго|медленн[оая])\s+(отвечают|ответ|доставка|поддержка)\b", "медленный сервис", t)
    t = re.sub(r"\b(нет|отсутствуют|пропал(и)?)\s+(детал(и|ей)|товар|части|комплектующ(ие|их))\b", "отсутствуют комплектующие", t)
    # Cleanup duplicated spaces
    t = re.sub(r"\s+", " ", t).strip()
    # Capitalize first letter for display
    if t:
        t = t[0].upper() + t[1:]
    return t
    problems = top_phrases(neg_texts)
    strengths = top_phrases(pos_texts)
    return {"problems": problems, "strengths": strengths}


def extract_key_insights(reviews: List[Review], max_items: int = 3) -> Dict[str, List[str]]:
    """Cluster positive/negative reviews and label clusters with representative keyphrases.

    Produces at most 2-3 concise, normalized multi-word insights for each polarity.
    """
    # Split by sentiment
    pos_docs = [r.text for r in reviews if (r.sentiment or "").startswith("pos")]
    neg_docs = [r.text for r in reviews if (r.sentiment or "").startswith("neg")]

    # Early exits
    if not pos_docs and not neg_docs:
        return {"problems": [], "strengths": []}

    # Prepare candidate phrases on full corpora to have a good pool
    def build_candidates(texts: List[str]) -> List[str]:
        if not texts:
            return []
        vec = TfidfVectorizer(
            max_features=12000,
            ngram_range=(2, 3),
            min_df=2,
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        try:
            X = vec.fit_transform([(t or "").lower() for t in texts])
        except ValueError:
            return []
        vocab = vec.get_feature_names_out()
        scores = X.mean(axis=0).A1
        pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
        cands: List[str] = []
        seen: set = set()
        for term, _ in pairs:
            norm = _normalize_phrase(term)
            if not norm or len(norm.split()) < 2:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            cands.append(norm)
            if len(cands) >= 200:
                break
        return cands

    pos_cands = build_candidates(pos_docs)
    neg_cands = build_candidates(neg_docs)

    # Load multilingual SBERT model once
    try:
        sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        # Fallback to english-only minimal model if multilingual unavailable
        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def cluster_and_label(texts: List[str], cands: List[str]) -> List[str]:
        if not texts:
            return []
        docs = [t for t in texts if t and t.strip()]
        if not docs:
            return []
        # Compute embeddings for docs
        doc_emb = sbert.encode(docs, normalize_embeddings=True)
        # Choose cluster count: 2 or 3 depending on size
        k = 2 if len(docs) < 30 else 3
        if len(docs) < 2:
            k = 1
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(doc_emb)
            centers = km.cluster_centers_
        except Exception:
            # If clustering fails, one cluster
            labels = np.zeros(len(docs), dtype=int)
            centers = np.mean(doc_emb, axis=0, keepdims=True)
            k = 1

        # Rank clusters by size
        sizes = [int(np.sum(labels == i)) for i in range(k)]
        order = np.argsort(sizes)[::-1]

        # Precompute candidate embeddings
        cand_list = cands[:200] if cands else []
        if not cand_list:
            # as a fallback, use normalized top phrases from texts via heuristic
            return extract_insights_heuristic([Review(text=t) for t in texts])["problems"][:max_items]
        cand_emb = sbert.encode(cand_list, normalize_embeddings=True)

        selected: List[str] = []
        seen: set = set()
        for idx in order:
            centroid = centers[idx:idx+1]
            sims = cosine_similarity(centroid, cand_emb)[0]
            ranked = np.argsort(sims)[::-1]
            # pick the first candidate that is not too similar to already selected
            for ridx in ranked[:30]:
                cand = cand_list[ridx]
                key = cand.lower()
                if key in seen:
                    continue
                # diversity against selected
                diverse = True
                for ex in selected:
                    w1 = set(cand.lower().split())
                    w2 = set(ex.lower().split())
                    if len(w1 & w2) / max(1, len(w1 | w2)) >= 0.6:
                        diverse = False
                        break
                if not diverse:
                    continue
                seen.add(key)
                selected.append(cand)
                break
            if len(selected) >= max_items:
                break
        return selected[:max_items]

    problems = cluster_and_label(neg_docs, neg_cands)
    strengths = cluster_and_label(pos_docs, pos_cands)

    # Final cleanup and limit to 2-3 items
    problems = _clean_insight_list(problems, polarity="negative")[:max_items]
    strengths = _clean_insight_list(strengths, polarity="positive")[:max_items]
    return {"problems": problems, "strengths": strengths}


def _clean_insight_list(items: List[str], polarity: str) -> List[str]:
    # Remove blanks, normalize, deduplicate, prefer multi-word phrases
    cleaned: List[str] = []
    seen: set = set()
    for raw in items:
        t = _strip_bullets(str(raw))
        t = re.sub(r"[\"'`\-•*]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        norm = t.lower()
        # Drop obvious junk
        junk = {"arrived", "really", "all", "item", "price", "poor", "bad"} if polarity == "negative" else {"good", "great", "nice", "price"}
        if norm in junk:
            continue
        norm = _normalize_phrase(norm)
        if not norm:
            continue
        # Require 2+ words after normalization
        if len(norm.split()) < 2:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(norm)
        if len(cleaned) >= 10:
            break
    return cleaned


def _strip_bullets(text: str) -> str:
    return (text or "").strip().lstrip("-•*\t ")


def generate_fake_reviews(n: int = 5) -> List[Review]:
    """
    @brief Генерирует список фиктивных отзывов для тестирования.

    @param n Количество отзывов для генерации (по умолчанию 5).
    @return Список объектов Review.

    @details
    Выбирает случайные примеры отзывов из предопределенного списка и создает для каждого
    уникальный идентификатор.
    """
    examples = [
        # English
        ("Delivery was late and the courier was rude", "negative", "Delivery"),
        ("Product is excellent, quality exceeded my expectations", "positive", "Quality"),
        ("Price is okay for the value", "neutral", "Price"),
        ("Very satisfied with the purchase, will buy again", "positive", "Satisfaction"),
        ("Package was damaged and the box was open", "negative", "Packaging"),
        ("Support answered quickly and helped solve my issue", "positive", "Support"),
        ("Battery life could be better", "neutral", "Battery"),
        # Russian
        ("Доставка опоздала на два дня", "negative", "Доставка"),
        ("Отличное качество товара, всем доволен", "positive", "Качество"),
        ("Цена нормальная, соответствует ожиданиям", "neutral", "Цена"),
        ("Очень доволен покупкой, рекомендую", "positive", "Удовлетворенность"),
        ("Упаковка была порвана, товар поцарапан", "negative", "Упаковка"),
        ("Поддержка ответила быстро и грамотно", "positive", "Поддержка"),
        ("Могла бы быть лучше автономность", "neutral", "Батарея"),
    ]
    reviews = []
    for _ in range(n):
        text, sentiment, topic = random.choice(examples)
        reviews.append(
            Review(
                id=str(uuid.uuid4()),
                text=text,
                sentiment=sentiment,
                sentiment_score=0.0,
                topic=topic,
                topic_id=None,
            )
        )
    return reviews


def detect_language(text: str) -> str:
    # very naive heuristic: Cyrillic -> ru, else en
    for ch in text:
        if "\u0400" <= ch <= "\u04FF":
            return "ru"
    return "en"


def simple_sentiment_score(text: str, language: str) -> Tuple[str, float]:
    positive_words_en = {
        "excellent",
        "good",
        "great",
        "satisfied",
        "recommend",
        "fast",
        "helped",
        "happy",
        "love",
    }
    negative_words_en = {
        "late",
        "bad",
        "poor",
        "damaged",
        "open",
        "rude",
        "slow",
        "issue",
        "broken",
    }

    positive_words_ru = {
        "отличное",
        "хорошо",
        "помог",
        "доволен",
        "рекомендую",
        "быстро",
        "супер",
    }
    negative_words_ru = {
        "плохо",
        "поздно",
        "опоздала",
        "порвана",
        "поцарапан",
        "проблема",
        "медленно",
    }

    text_lower = text.lower()
    if language == "ru":
        pos = sum(word in text_lower for word in positive_words_ru)
        neg = sum(word in text_lower for word in negative_words_ru)
    else:
        pos = sum(word in text_lower for word in positive_words_en)
        neg = sum(word in text_lower for word in negative_words_en)

    # Smooth fractional score in [-1, 1] to avoid integer-looking values
    raw = float(pos - neg)
    denom = float(max(pos + neg, 1))
    score = float(np.tanh(raw / denom))
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    return label, score


def extract_topics(texts: List[str], reviews: List[Review] = None, n_topics: int = 5, n_keywords: int = 5) -> Tuple[np.ndarray, Dict[int, Dict]]:
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < n_topics:
        n_topics = max(2, min(X.shape[0], 5))
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    doc_topic = svd.fit_transform(X)
    doc_topic = normalize(doc_topic)
    topic_terms = svd.components_
    vocab = np.array(vectorizer.get_feature_names_out())
    topics_info: Dict[int, Dict] = {}
    
    # Create topics with meaningful names
    for k in range(topic_terms.shape[0]):
        top_idx = np.argsort(topic_terms[k])[::-1][:n_keywords]
        keywords = vocab[top_idx].tolist()
        
        # Get dominant sentiment for this topic
        topic_sentiment = "neutral"
        if reviews:
            # Find documents assigned to this topic
            doc_assignments = np.argmax(doc_topic, axis=1)
            topic_docs = [i for i, t in enumerate(doc_assignments) if t == k]
            if topic_docs:
                sentiments = [reviews[i].sentiment for i in topic_docs if i < len(reviews)]
                if sentiments:
                    # Use most common sentiment
                    from collections import Counter
                    sentiment_counts = Counter(sentiments)
                    topic_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Create meaningful topic name
        topic_name = _create_topic_name(keywords, topic_sentiment)
        
        # Skip topics that don't have clear concepts
        if topic_name is None:
            continue
            
        topics_info[k] = {"label": topic_name, "keywords": keywords, "size": 0}
    
    # Filter out empty topics and reassign
    valid_topics = {k: v for k, v in topics_info.items() if v["label"] is not None}
    if not valid_topics:
        # Fallback: create one general topic
        valid_topics = {0: {"label": "Общее", "keywords": ["general"], "size": 0}}
    
    # Reassign documents to valid topics only
    assignments = np.argmax(doc_topic, axis=1)
    filtered_assignments = []
    topic_id_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_topics.keys())}
    
    for t in assignments:
        if t in valid_topics:
            filtered_assignments.append(topic_id_mapping[t])
            valid_topics[t]["size"] += 1
        else:
            # Assign to first valid topic if original topic was filtered out
            first_valid = list(valid_topics.keys())[0]
            filtered_assignments.append(topic_id_mapping[first_valid])
            valid_topics[first_valid]["size"] += 1
    
    # Renumber topics to be consecutive
    final_topics = {}
    for i, (old_id, topic_data) in enumerate(valid_topics.items()):
        final_topics[i] = topic_data
    
    return np.array(filtered_assignments), final_topics


def _ru_negative_heuristic(text: str) -> bool:
    """Very simple Russian negativity detector for cases models miss profanity/strong negatives."""
    t = (text or "").lower()
    negative_stems = [
        "ужас", "отстой", "мерз", "кошмар", "ненавиж", "разочар", "плох", "дурн", "говн",
        "обман", "мошен", "хлам", "катастр", "не совет", "не рекоменд", "возмут",
    ]
    # Flip if strong negative phrase or many exclamations
    if t.count("!") >= 2:
        return True
    for stem in negative_stems:
        if stem in t:
            return True
    return False


def _create_topic_name(keywords: List[str], sentiment: str = "neutral") -> str:
    """Create a meaningful topic name from keywords and sentiment"""
    if not keywords:
        return "Общее"
    
    # Take top keywords
    top_keywords = [k.lower() for k in keywords[:4]]
    
    # Sentiment prefixes
    sentiment_prefixes = {
        "positive": {"ru": "Хорошее", "en": "Good"},
        "negative": {"ru": "Плохое", "en": "Poor"}, 
        "neutral": {"ru": "", "en": ""}
    }
    
    # Core topic mappings with sentiment-aware names
    topic_concepts = {
        # Delivery/Shipping
        "delivery": {"ru": "доставка", "en": "delivery"},
        "shipping": {"ru": "доставка", "en": "delivery"},
        "fast": {"ru": "быстрая доставка", "en": "fast delivery"},
        "slow": {"ru": "медленная доставка", "en": "slow delivery"},
        "late": {"ru": "опоздание", "en": "late delivery"},
        
        # Quality
        "quality": {"ru": "качество", "en": "quality"},
        "good": {"ru": "качество", "en": "quality"},
        "excellent": {"ru": "качество", "en": "quality"},
        "poor": {"ru": "качество", "en": "quality"},
        "bad": {"ru": "качество", "en": "quality"},
        
        # Price
        "price": {"ru": "цена", "en": "price"},
        "cheap": {"ru": "цена", "en": "price"},
        "expensive": {"ru": "цена", "en": "price"},
        "cost": {"ru": "цена", "en": "price"},
        
        # Service/Support
        "service": {"ru": "сервис", "en": "service"},
        "support": {"ru": "поддержка", "en": "support"},
        "help": {"ru": "поддержка", "en": "support"},
        "customer": {"ru": "сервис", "en": "service"},
        
        # Product/Item
        "product": {"ru": "товар", "en": "product"},
        "item": {"ru": "товар", "en": "product"},
        "package": {"ru": "упаковка", "en": "packaging"},
        "box": {"ru": "упаковка", "en": "packaging"},
        
        # Technical
        "battery": {"ru": "батарея", "en": "battery"},
        "life": {"ru": "автономность", "en": "battery life"},
        "design": {"ru": "дизайн", "en": "design"},
        "look": {"ru": "внешний вид", "en": "appearance"},
        "easy": {"ru": "удобство", "en": "usability"},
        "simple": {"ru": "простота", "en": "simplicity"},
        
        # Russian specific
        "доставка": {"ru": "доставка", "en": "delivery"},
        "быстро": {"ru": "быстрая доставка", "en": "fast delivery"},
        "медленно": {"ru": "медленная доставка", "en": "slow delivery"},
        "качество": {"ru": "качество", "en": "quality"},
        "хорошо": {"ru": "качество", "en": "quality"},
        "плохо": {"ru": "качество", "en": "quality"},
        "цена": {"ru": "цена", "en": "price"},
        "дешево": {"ru": "цена", "en": "price"},
        "дорого": {"ru": "цена", "en": "price"},
        "сервис": {"ru": "сервис", "en": "service"},
        "поддержка": {"ru": "поддержка", "en": "support"},
        "товар": {"ru": "товар", "en": "product"},
        "упаковка": {"ru": "упаковка", "en": "packaging"},
        "батарея": {"ru": "батарея", "en": "battery"},
        "дизайн": {"ru": "дизайн", "en": "design"},
        "удобно": {"ru": "удобство", "en": "usability"},
        "просто": {"ru": "простота", "en": "simplicity"},
    }
    
    # Find the best matching concept
    best_concept = None
    for keyword in top_keywords:
        if keyword in topic_concepts:
            best_concept = topic_concepts[keyword]
            break
    
    if best_concept:
        # Determine language (simple heuristic)
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in ' '.join(keywords))
        lang = "ru" if has_cyrillic else "en"
        
        concept_name = best_concept[lang]
        prefix = sentiment_prefixes[sentiment][lang]
        
        if prefix:
            return f"{prefix} {concept_name.title()}"
        else:
            return concept_name.title()
    
    # Fallback: if no clear concept, skip this topic
    return None


def _normalize_hf_label(raw_label: str) -> str:
    label = str(raw_label).strip().lower()
    # common patterns: 'positive', 'negative', 'neutral'
    if label.startswith("pos"):
        return "positive"
    if label.startswith("neg"):
        return "negative"
    if label.startswith("neu"):
        return "neutral"
    # star patterns: '1 star', '2 stars', '5 stars'
    m = re.search(r"(\d)", label)
    if m:
        stars = int(m.group(1))
        if stars >= 4:
            return "positive"
        if stars <= 2:
            return "negative"
        return "neutral"
    # label_x patterns
    m2 = re.search(r"label[_\s-]?(\d)", label)
    if m2:
        val = int(m2.group(1))
        if val >= 4:
            return "positive"
        if val <= 2:
            return "negative"
        return "neutral"
    return "neutral"


def _hf_sentiment_batch(texts: List[str], model: str, token: str) -> List[Tuple[str, float]]:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    # HF supports sending list of inputs
    backoffs = [1, 2, 4]
    last_exc = None
    for attempt, delay in enumerate([0] + backoffs):
        if delay:
            try:
                import time as _time
                _time.sleep(delay)
            except Exception:
                pass
        try:
            response = requests.post(url, headers=headers, json={"inputs": texts}, timeout=60)
            status = response.status_code
            try:
                data = response.json()
            except Exception:
                data = {"non_json": response.text}
            response.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            log_entry = {
                "url": url,
                "status": locals().get("status", None),
                "inputs_count": len(texts),
                "error": str(e),
                "response": locals().get("data", None),
            }
            _HF_API_LOGS.insert(0, log_entry)
            if len(_HF_API_LOGS) > 50:
                del _HF_API_LOGS[50:]
            print("[HF DEBUG ERROR]", log_entry)
            if attempt == len(backoffs):
                raise
    # store success log entry at the top
    log_entry = {
        "url": url,
        "status": status,
        "inputs_count": len(texts),
        "response": data,
    }
    _HF_API_LOGS.insert(0, log_entry)
    if len(_HF_API_LOGS) > 50:
        del _HF_API_LOGS[50:]
    print("[HF DEBUG]", log_entry)
    results: List[Tuple[str, float]] = []
    # data may be list[dict] or list[list[dict]] depending on model
    for item in data:
        if isinstance(item, list):
            # pick max score
            best = max(item, key=lambda x: x.get("score", 0))
            label = str(best.get("label", "neutral")).lower()
            score = float(best.get("score", 0.0))
        elif isinstance(item, dict):
            label = str(item.get("label", "neutral")).lower()
            score = float(item.get("score", 0.0))
        else:
            label, score = "neutral", 0.0
        label = _normalize_hf_label(label)
        results.append((label, score))
    
    # Fix: if we got only 1 result but sent multiple texts, the API returned nested structure
    if len(results) == 1 and len(texts) > 1:
        # The single result contains all predictions
        single_result = data[0]
        if isinstance(single_result, list):
            results = []
            for pred in single_result:
                label = str(pred.get("label", "neutral")).lower()
                score = float(pred.get("score", 0.0))
                label = _normalize_hf_label(label)
                results.append((label, score))
    return results


def analyze_sentiment_ai(
    reviews: List[Review],
    model: str,
    token: str,
    batch_size: int = 16,
) -> None:
    # Auto routing by language to stronger models
    if model.strip().lower().startswith("auto"):
        # Use lxyuan as default for RU and others as it's robust multilingual
        ru_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        en_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        multi_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        # RU batch
        ru_indices = [i for i, r in enumerate(reviews) if (r.language or "").startswith("ru")]
        if ru_indices:
            texts = [reviews[i].text for i in ru_indices]
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                try:
                    preds = _hf_sentiment_batch(batch, model=ru_model, token=token)
                except Exception:
                    # Fallback to heuristic scoring
                    preds = []
                    for j in range(len(batch)):
                        r_fallback = reviews[ru_indices[start + j]]
                        label_fb, score_fb = simple_sentiment_score(r_fallback.text, r_fallback.language or "en")
                        preds.append((label_fb, score_fb))
                for j, (label, score) in enumerate(preds):
                    r = reviews[ru_indices[start + j]]
                    # Heuristic: fix obviously negative/profane Russian misclassifications
                    if _ru_negative_heuristic(r.text) and label == "positive" and score < 0.9:
                        label, score = "negative", max(score, 0.6)
                    r.sentiment = label
                    r.sentiment_score = score
        # Non-RU batch split into EN and other
        non_ru_indices = [i for i, r in enumerate(reviews) if i not in ru_indices]
        en_indices = [i for i in non_ru_indices if (reviews[i].language or "").startswith("en")]
        other_indices = [i for i in non_ru_indices if i not in en_indices]
        if en_indices:
            texts = [reviews[i].text for i in en_indices]
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                try:
                    preds = _hf_sentiment_batch(batch, model=en_model, token=token)
                except Exception:
                    preds = []
                    for j in range(len(batch)):
                        r_fallback = reviews[en_indices[start + j]]
                        label_fb, score_fb = simple_sentiment_score(r_fallback.text, r_fallback.language or "en")
                        preds.append((label_fb, score_fb))
                for j, (label, score) in enumerate(preds):
                    r = reviews[en_indices[start + j]]
                    r.sentiment = label
                    r.sentiment_score = score
        if other_indices:
            texts = [reviews[i].text for i in other_indices]
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                try:
                    preds = _hf_sentiment_batch(batch, model=multi_model, token=token)
                except Exception:
                    preds = []
                    for j in range(len(batch)):
                        r_fallback = reviews[other_indices[start + j]]
                        label_fb, score_fb = simple_sentiment_score(r_fallback.text, r_fallback.language or "en")
                        preds.append((label_fb, score_fb))
                for j, (label, score) in enumerate(preds):
                    r = reviews[other_indices[start + j]]
                    r.sentiment = label
                    r.sentiment_score = score
        return

    # Single model path
    texts = [r.text for r in reviews]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            preds = _hf_sentiment_batch(batch, model=model, token=token)
        except Exception:
            # Fallback to heuristic scoring instead of leaving zeros
            preds = []
            for r_fb in reviews[start:start + batch_size]:
                label_fb, score_fb = simple_sentiment_score(r_fb.text, r_fb.language or "en")
                preds.append((label_fb, score_fb))
        for r, (label, score) in zip(reviews[start:start + batch_size], preds):
            r.sentiment = label
            r.sentiment_score = score


def run_analysis(
    reviews: List[Review],
    use_ai: bool = False,
    hf_model: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Analysis:
    """
    @brief Выполняет простой анализ списка отзывов, подсчитывая количество отзывов по настроению.

    @param reviews Список объектов Review для анализа.
    @return Объект Analysis с уникальным идентификатором, датой создания, исходными отзывами и статистикой.

    @details
    Статистика включает количество отзывов с позитивным, негативным и нейтральным настроением.
    """
    # Language detection
    for r in reviews:
        r.language = detect_language(r.text)

    # Sentiment: AI if configured, else rule-based
    if use_ai and hf_token and hf_model:
        analyze_sentiment_ai(reviews, model=hf_model, token=hf_token)
    else:
        for r in reviews:
            label, score = simple_sentiment_score(r.text, r.language or "en")
            r.sentiment = label
            r.sentiment_score = score

    # Topic extraction
    texts = [r.text for r in reviews]
    topic_ids, topics_info = extract_topics(texts, reviews=reviews, n_topics=min(6, max(2, len(reviews)//2)), n_keywords=6)
    for r, tid in zip(reviews, topic_ids):
        r.topic_id = int(tid)
        r.topic = topics_info[int(tid)]["label"]

    # Stats aggregation
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    topic_distribution: Dict[str, int] = {}
    for r in reviews:
        normalized = r.sentiment if r.sentiment in sentiment_counts else _normalize_hf_label(r.sentiment)
        if normalized not in sentiment_counts:
            normalized = "neutral"
        r.sentiment = normalized
        sentiment_counts[normalized] += 1
        topic_distribution[r.topic or "Unknown"] = topic_distribution.get(r.topic or "Unknown", 0) + 1

    # Insights (problems / strengths): prefer embedding-based clustering, then AI, then heuristic
    insights = extract_key_insights(reviews, max_items=3)
    # Clarify labels via rule-based and optional LLM rewrite
    insights = clarify_insights(reviews, insights, hf_token if use_ai and hf_token else None)
    if not insights["problems"] and not insights["strengths"] and use_ai and hf_token:
        insights = extract_insights_ai(reviews, token=hf_token)
        # Reduce to 3 and clean again
        insights = {
            "problems": _clean_insight_list(insights.get("problems", []), polarity="negative")[:3],
            "strengths": _clean_insight_list(insights.get("strengths", []), polarity="positive")[:3],
        }
    if not insights["problems"] and not insights["strengths"]:
        insights = extract_insights_heuristic(reviews)
        insights = {
            "problems": _clean_insight_list(insights.get("problems", []), polarity="negative")[:3],
            "strengths": _clean_insight_list(insights.get("strengths", []), polarity="positive")[:3],
        }

    stats = {
        "sentiment_counts": sentiment_counts,
        "topic_distribution": topic_distribution,
        "topics": topics_info,
        "insights": insights,
    }

    return Analysis(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        reviews=reviews,
        stats=stats,
    )


def load_reviews_from_file(file_bytes: bytes, file_name: str) -> List[Review]:
    name = file_name.lower()
    if name.endswith(".csv") or name.endswith(".tsv"):
        sep = "," if name.endswith(".csv") else "\t"
        # Try different encodings for Russian text
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(pd.io.common.BytesIO(file_bytes), sep=sep, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not decode file with any supported encoding")
    elif name.endswith(".xlsx"):
        df = pd.read_excel(pd.io.common.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file format. Use CSV, TSV, or XLSX.")

    text_col = None
    for cand in ["text", "review", "comment", "Отзыв", "текст", "message"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        # fallback: first column
        text_col = df.columns[0]

    texts = df[text_col].astype(str).fillna("").tolist()
    return [Review(text=t) for t in texts if t.strip()]
