from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import uuid


@dataclass
class Review:
    """
    @brief Класс, представляющий отдельный отзыв пользователя.

    @var id Уникальный идентификатор отзыва.
    @var text Текст отзыва.
    @var sentiment Настроение отзыва: "positive", "negative", "neutral".
    @var topic Тема отзыва (необязательное поле).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    sentiment: str = ""  # "positive", "negative", "neutral"
    sentiment_score: float = 0.0
    topic: Optional[str] = None
    topic_id: Optional[int] = None
    language: Optional[str] = None  # ru, en, etc.


@dataclass
class ReviewFile:
    """
    @brief Класс, представляющий файл с отзывами.

    @var id Уникальный идентификатор файла.
    @var name Имя файла.
    @var format Формат файла: CSV, TSV, XLSX.
    @var uploaded_at Дата и время загрузки файла.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    format: str = ""  # CSV, TSV, XLSX
    uploaded_at: datetime = field(default_factory=datetime.now)


@dataclass
class Analysis:
    """
    @brief Класс, представляющий анализ отзывов.

    @var id Уникальный идентификатор анализа.
    @var created_at Дата и время создания анализа.
    @var reviews Список объектов Review, включенных в анализ.
    @var stats Статистика по анализу (количество positive, negative, neutral).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    reviews: List[Review] = field(default_factory=list)
    stats: Dict[str, Dict] = field(
        default_factory=lambda: {
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "topic_distribution": {},
            "topics": {},  # id -> {label, keywords, size}
        }
    )


@dataclass
class User:
    """
    @brief Класс, представляющий пользователя системы.

    @var id Уникальный идентификатор пользователя.
    @var name Имя пользователя.
    @var email Электронная почта пользователя.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
