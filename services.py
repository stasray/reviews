import random
import uuid
from datetime import datetime
from typing import List
from models import Review, Analysis


def generate_fake_reviews(n: int = 5) -> List[Review]:
    examples = [
        ("Delivery was late", "negative", "Delivery"),
        ("Product is excellent", "positive", "Quality"),
        ("Price is okay", "neutral", "Price"),
        ("Very satisfied", "positive", "General"),
        ("Package was damaged", "negative", "Packaging"),
    ]
    reviews = []
    for _ in range(n):
        text, sentiment, topic = random.choice(examples)
        reviews.append(
            Review(id=str(uuid.uuid4()), text=text, sentiment=sentiment, topic=topic)
        )
    return reviews


def run_analysis(reviews: List[Review]) -> Analysis:
    stats = {"positive": 0, "negative": 0, "neutral": 0}
    for r in reviews:
        stats[r.sentiment] += 1
    return Analysis(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        reviews=reviews,
        stats=stats,
    )
