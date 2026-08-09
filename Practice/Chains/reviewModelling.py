from pydantic import BaseModel, Field
from enum import Enum

class ReviewAnalysisSecurity(BaseModel):
    review: str

    securityCheck: bool = Field(
        description="""Check the review for security:

        Security Checks:
        - Review does not contain sexual comments
        - Review does not contain PII
        - Review is about a product and nothing else

        If any check fails, return False.
        Otherwise return True.
        """
    )


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
    newReview = ReviewAnalysisSecurity(
        review="Very Nice Product",
        securityCheck=True
    )

    print(newReview)
    print(newReview.model_dump())