from typing import TypedDict ,Annotated ,Optional , Literal
from enum import Enum


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(TypedDict):
    review: str
    sentiment: Sentiment
    emotion: Annotated[str,"Tel the sentiment of the user , what it feels from the review"]
    complaint: str


if __name__ == "__main__":

    newReview: ReviewAnalysis = {
        "review": "Very Nice Product",
        "sentiment": Sentiment.POSITIVE,
        "emotion": "happy",
        "complaint": ""
    }

    print(newReview)