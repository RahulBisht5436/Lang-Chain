from pydantic import BaseModel, Field
from enum import Enum


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    review: str

    sentiment: Sentiment

    emotion: str = Field(
        description="Tell the sentiment of the user, what they feel from the review"
    )

    complaint: str


if __name__ == "__main__":

    newReview = ReviewAnalysis(
        review="Very Nice Product",
        sentiment=Sentiment.POSITIVE,
        emotion="happy",
        complaint=""
    )

    print(newReview)
    print(newReview.model_dump())
