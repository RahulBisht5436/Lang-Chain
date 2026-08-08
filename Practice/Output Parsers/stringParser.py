import os
import sys
from pathlib import Path

# Import StrOutputParser
# It converts the AIMessage returned by a chat model
# into a simple Python string.
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate


# Add the parent directory to Python's module search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.openAI_llm import llm


# ---------------------------------------------------------
# 1. Detailed Answer Prompt
# ---------------------------------------------------------

template_detailed = ChatPromptTemplate(
    [
        (
            "system",
            "You are expert in the field. "
            "Tell in detail about the topic: {topic}",
        )
    ]
)


# ---------------------------------------------------------
# 2. Summary Prompt
# ---------------------------------------------------------

template_summary = ChatPromptTemplate(
    [
        (
            "human",
            "Summarize this data: {data} in 5 pointers",
        )
    ]
)


# ---------------------------------------------------------
# 3. Create the Output Parser
# ---------------------------------------------------------

# StrOutputParser converts the LLM's response:
#
#     AIMessage(content="Python is a programming language...")
#
# into:
#
#     "Python is a programming language..."
#
# In other words:
#
# AIMessage  --->  String
#
str_parser = StrOutputParser()


# ---------------------------------------------------------
# 4. Create Detailed Answer Chain
# ---------------------------------------------------------

# Without parser:
#
# template_detailed | llm
#
# The output will normally be an AIMessage object.
#
# With parser:
#
# template_detailed | llm | str_parser
#
# The output will be a plain Python string.

chainDetailed = template_detailed | llm | str_parser


# ---------------------------------------------------------
# 5. Create Summary Chain
# ---------------------------------------------------------

# Again, the StrOutputParser converts the AIMessage
# returned by the LLM into a normal Python string.

chainSummary = template_summary | llm | str_parser


# ---------------------------------------------------------
# 6. Take User Input
# ---------------------------------------------------------

ques = input("What do you want to ask:\n")


if ques:

    # -----------------------------------------------------
    # 7. Generate Detailed Answer
    # -----------------------------------------------------

    # Because StrOutputParser is part of chainDetailed,
    # detailedAnswer will be a STRING.
    #
    # Without StrOutputParser:
    #
    # detailedAnswer = AIMessage(...)
    #
    # With StrOutputParser:
    #
    # detailedAnswer = "Python is a programming language..."
    #
    detailedAnswer = chainDetailed.invoke(
        {"topic": ques}
    )

    print("\n================ DETAILED ANSWER ================\n")
    print(detailedAnswer)


    # -----------------------------------------------------
    # 8. Send Detailed Answer to Summary Chain
    # -----------------------------------------------------

    # detailedAnswer is already a string because
    # StrOutputParser processed the first LLM response.
    #
    # Therefore it can easily be passed as {data}.
    #
    summaryAnswer = chainSummary.invoke(
        {"data": detailedAnswer}
    )


    # -----------------------------------------------------
    # 9. Print Summary
    # -----------------------------------------------------

    print("\n=========================>>>>>>\n")
    print(summaryAnswer)