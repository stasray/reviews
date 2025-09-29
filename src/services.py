"""
Основной модуль сервисов для анализа отзывов.

Содержит главные функции для анализа отзывов, используя
разделенные модули для различных аспектов анализа.
"""

import random
import uuid
from datetime import datetime
from typing import List, Optional

from models import Review, Analysis

# Import from our modularized services
from sentiment_analysis import (
    detect_language,
    simple_sentiment_score,
    analyze_sentiment_ai,
    _normalize_hf_label
)
from topic_modeling import extract_topics
from insights_extraction import (
    extract_insights_heuristic,
    extract_key_insights,
    clarify_insights
)
from huggingface_client import extract_insights_ai
from file_processing import load_reviews_from_file
from text_utils import _clean_insight_list


def generate_fake_reviews(n: int = 5) -> List[Review]:
    """
    Генерирует список фиктивных отзывов для тестирования.

    Args:
        n: Количество отзывов для генерации (по умолчанию 5).
        
    Returns:
        Список объектов Review.
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


def run_analysis(
    reviews: List[Review],
    use_ai: bool = False,
    hf_model: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Analysis:
    """
    Выполняет анализ списка отзывов.

    Args:
        reviews: Список объектов Review для анализа.
        use_ai: Использовать ли AI для анализа тональности.
        hf_model: Название модели Hugging Face.
        hf_token: API токен Hugging Face.
        
    Returns:
        Объект Analysis с результатами анализа.
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
    topic_ids, topics_info = extract_topics(
        texts, 
        reviews=reviews, 
        n_topics=min(6, max(2, len(reviews)//2)), 
        n_keywords=6
    )
    
    for r, tid in zip(reviews, topic_ids):
        r.topic_id = int(tid)
        r.topic = topics_info[int(tid)]["label"]

    # Stats aggregation
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    topic_distribution: dict = {}
    
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