import os
import sys
from pathlib import Path

from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# Add parent directory to Python path
# ---------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------
# Import main LLM
# ---------------------------------------------------------

from llm.huggingFaceUnstructured_llm import llm


# =========================================================
# 1. Schema Generation
# =========================================================

schema_details = [
    ResponseSchema(name="Topic", description="Name of the topic"),
    ResponseSchema(
        name="Summary", description="A short summary of approximately 50 words"
    ),
    ResponseSchema(
        name="Details", description="A detailed description of approximately 200 words"
    ),
    ResponseSchema(
        name="Main_ideas", description="An array/list containing the main ideas"
    ),
]


schema_summary = [
    ResponseSchema(name="Topic", description="Name of the topic"),
    ResponseSchema(
        name="Summary", description="A short summary of approximately 50 words"
    ),
    ResponseSchema(
        name="Main_ideas", description="An array/list containing the main ideas"
    ),
    ResponseSchema(
        name="tech_Stack",
        description="An array/list of technologies that need to be used",
    ),
]


# =========================================================
# 2. Create Structured Output Parsers
# =========================================================

parser_detailed = StructuredOutputParser.from_response_schemas(schema_details)

parser_summary = StructuredOutputParser.from_response_schemas(schema_summary)


# =========================================================
# 3. Get Format Instructions
# =========================================================

detailed_format_instructions = parser_detailed.get_format_instructions()

summary_format_instructions = parser_summary.get_format_instructions()


# =========================================================
# 4. Prompt Templates
# =========================================================

template_detailed = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert in the given field.

Explain the following topic in detail:

{topic}

You MUST follow these formatting instructions:

{format_instructions}
""",
        )
    ]
)


template_summary = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Summarize the following data:

{data}

Provide the result according to these formatting instructions:

{format_instructions}
""",
        )
    ]
)


# =========================================================
# 5. Create Chains
# =========================================================

chainDetailed = template_detailed | llm | parser_detailed


chainSummary = template_summary | llm | parser_summary


# =========================================================
# 6. Application
# =========================================================

ques = input("What do you want to ask:\n")


if ques:

    # -----------------------------------------------------
    # Generate detailed answer
    # -----------------------------------------------------

    detailedAnswer = chainDetailed.invoke(
        {"topic": ques, "format_instructions": detailed_format_instructions}
    )

    print("\n================ DETAILED ANSWER ================\n")

    print(detailedAnswer)

    # -----------------------------------------------------
    # Generate summary
    # -----------------------------------------------------

    summaryAnswer = chainSummary.invoke(
        {"data": detailedAnswer, "format_instructions": summary_format_instructions}
    )

    print("\n================ SUMMARY ================\n")

    print(summaryAnswer)
