"""
Пакет для анализа отзывов.

Содержит модули для анализа тональности, извлечения тем,
обработки файлов и других функций анализа отзывов.
"""

from .sentiment_analysis import (
    detect_language,
    simple_sentiment_score,
    analyze_sentiment_ai,
    get_hf_logs
)

from .topic_modeling import extract_topics

from .insights_extraction import (
    extract_insights_heuristic,
    extract_key_insights,
    clarify_insights
)

from .file_processing import load_reviews_from_file

from .huggingface_client import extract_insights_ai

from .text_utils import (
    _normalize_phrase,
    _strip_bullets,
    _clean_insight_list,
    get_stopwords
)

__all__ = [
    # Sentiment analysis
    'detect_language',
    'simple_sentiment_score', 
    'analyze_sentiment_ai',
    'get_hf_logs',
    
    # Topic modeling
    'extract_topics',
    
    # Insights extraction
    'extract_insights_heuristic',
    'extract_key_insights',
    'clarify_insights',
    'extract_insights_ai',
    
    # File processing
    'load_reviews_from_file',
    
    # Text utilities
    '_normalize_phrase',
    '_strip_bullets',
    '_clean_insight_list',
    'get_stopwords'
]

