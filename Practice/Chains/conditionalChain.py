import os
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate

from reviewModelling import ReviewAnalysisSecurity, ReviewAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.openAI_llm import llm


review = input("Enter the review: ")

parser = StrOutputParser()


# ---------------------------------------------------------
# 1. Security Prompt
# ---------------------------------------------------------

checkSecurityPrompt = PromptTemplate(
    template="""
Check the following review for security:

Review:
{review}

Security Checks:
1. The review must not contain sexual comments.
2. The review must not contain PII.
3. The review must be about a product.

If ANY security check fails, return False.
Otherwise return True.
""",
    input_variables=["review"],
)


# ---------------------------------------------------------
# 2. Sentiment Response Prompts
# ---------------------------------------------------------

sentimentPositivePrompt = PromptTemplate(
    template="""
You are a customer support assistant.

The customer has given a positive review about:
{topic}

Thank the customer for their positive feedback and politely ask them
to leave a review/rating for the product.

Keep the response short, friendly, and professional.
""",
    input_variables=["topic"],
)


sentimentNegativePrompt = PromptTemplate(
    template="""
You are a customer support assistant.

The customer has given a negative review about:
{topic}

Apologize sincerely for the customer's experience.

Ask them to contact our support team at +91-XXXXXXXXXX so that we can
understand the issue and help resolve it.

Keep the response empathetic, professional, and helpful.
""",
    input_variables=["topic"],
)


sentimentNeutralPrompt = PromptTemplate(
    template="""
You are a customer support assistant.

The customer has given a neutral review about:
{topic}

Thank them for their feedback and recommend a few relevant products
or product options they may be interested in.

Also mention that they can check our latest offers and deals.

Keep the response helpful, concise, and professional.
""",
    input_variables=["topic"],
)


# ---------------------------------------------------------
# 3. Structured LLMs
# ---------------------------------------------------------

checkSecurityLLM = llm.with_structured_output(
    ReviewAnalysisSecurity
)

checkSentimentLLM = llm.with_structured_output(
    ReviewAnalysis
)


# ---------------------------------------------------------
# 4. Security Check
# ---------------------------------------------------------

if review:

    answer = checkSecurityLLM.invoke(
        checkSecurityPrompt.invoke({"review": review})
    )

    print("Security Analysis:")
    print(answer)


    # -----------------------------------------------------
    # 5. Only process safe reviews
    # -----------------------------------------------------

    if answer.securityCheck:

        reviewAnalysis = checkSentimentLLM.invoke(review)

        print("\nSentiment Analysis:")
        print(reviewAnalysis)


        # -------------------------------------------------
        # 6. Conditional Branch
        # -------------------------------------------------

        branchConditionalChain = RunnableBranch(

            (
                 lambda x: x["sentiment"].lower() == "positive",

                sentimentPositivePrompt
                | llm
                | parser
            ),

            (
                 lambda x: x["sentiment"].lower() == "negative",

                sentimentNegativePrompt
                | llm
                | parser
            ),

            sentimentNeutralPrompt
            | llm
            | parser
        )


        # -------------------------------------------------
        # 7. Execute branch
        # -------------------------------------------------
        branchConditionalChain.get_graph().print_ascii()
        output = branchConditionalChain.invoke(
            {
                "sentiment": reviewAnalysis.sentiment,
                "topic": review
            }
        )

        print("\nFinal Response:")
        print(output)

    else:

        print("Inappropriate Comment, Can't process your request.")

else:

    print("Please enter a review.")