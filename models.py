from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import uuid


@dataclass
class Review:
    id: str
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    topic: str = None


@dataclass
class ReviewFile:
    id: str
    name: str
    format: str  # CSV, TSV, XLSX
    uploaded_at: datetime


@dataclass
class Analysis:
    id: str
    created_at: datetime
    reviews: List[Review] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class User:
    id: str
    name: str
    email: str
