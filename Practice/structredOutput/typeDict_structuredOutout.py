from typing import TypedDict
from enum import Enum


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(TypedDict):
    review: str
    sentiment: Sentiment
    emotion: str
    complaint: str


if __name__ == "__main__":

    newReview: ReviewAnalysis = {
        "review": "Very Nice Product",
        "sentiment": Sentiment.POSITIVE,
        "emotion": "happy",
        "complaint": ""
    }

    print(newReview)