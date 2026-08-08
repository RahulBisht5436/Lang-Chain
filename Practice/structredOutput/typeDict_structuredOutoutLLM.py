import os
import sys
from pathlib import Path

# Import the Pydantic/TypedDict schema that defines
# the structure expected from the LLM's output.
from typeDict_structuredOutout import ReviewAnalysis

import streamlit as st
from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. HANDLE PROJECT PATH
# ============================================================

# __file__ represents the current Python file.
#
# Path(__file__).resolve()
#     -> Converts the current file path into an absolute path.
#
# .parents[1]
#     -> Goes two levels up in the directory structure.
#
# Example:
# C:/project/Practice/app/file.py
#             ↓
# C:/project/Practice
#
# This is useful when importing modules from another directory
# in the project.
current_dir = Path(__file__).resolve().parents[1]

# Add the project root directory to Python's module search path.
#
# This allows Python to find modules such as:
#     from llm.openAI_llm import llm
sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT THE LLM
# ============================================================

# Import the LLM configuration/object from our project.
#
# The actual LLM configuration is kept in a separate file:
#     llm/openAI_llm.py
#
# This keeps the LLM configuration separate from our application
# logic.
from llm.openAI_llm import llm


# ============================================================
# 3. CREATE THE PROMPT TEMPLATE
# ============================================================

# PromptTemplate is used to create a reusable prompt.
#
# {review} is a placeholder that will be replaced with the
# user's actual review at runtime.
template = PromptTemplate(
    template="""
We need a structured output for the review:

{review}

The output should have this structure:

{{
    "review": "Very Nice Product",
    "sentiment": "positive",
    "emotion": "happy",
    "complaint": ""
}}
""",

    # Tell LangChain which variables exist inside the template.
    #
    # Since our prompt contains {review}, we must specify
    # "review" here.
    input_variables=["review"],
)


# ============================================================
# 4. CONVERT NORMAL LLM INTO STRUCTURED-OUTPUT LLM
# ============================================================

# ReviewAnalysis defines the structure/schema that we expect
# from the LLM.
#
# with_structured_output() tells LangChain:
#
# "Do not simply return normal text.
#  Return the response according to the ReviewAnalysis schema."
#
# This is one of the important features of structured output
# in LangChain.
structured_llm = llm.with_structured_output(ReviewAnalysis)


# ============================================================
# 5. CREATE THE LANGCHAIN CHAIN
# ============================================================

# The "|" operator creates a LangChain RunnableSequence.
#
# Flow:
#
# User Review
#      ↓
# PromptTemplate
#      ↓
# Structured LLM
#      ↓
# ReviewAnalysis object
#
# So:
#
# template | structured_llm
#
# means:
# First execute the prompt template,
# then send the generated prompt to the structured LLM.
chain = template | structured_llm


# ============================================================
# 6. STREAMLIT USER INTERFACE
# ============================================================

# Display the main heading of the Streamlit application.
st.header("Sentiment Analysis")


# Create a text input box where the user can enter
# a product review.
#
# Example:
# "The product is amazing. I really loved it!"
review = st.text_input("Enter the review")


# Create an "Analyse Review" button.
#
# The code inside the if block executes only when
# the user clicks this button.
if st.button("Analyse Review"):

    # Send the user's review into the LangChain chain.
    #
    # The dictionary key "review" must match the
    # input_variables defined in PromptTemplate.
    #
    # Internally:
    #
    # {"review": review}
    #          ↓
    # PromptTemplate
    #          ↓
    # Generated prompt
    #          ↓
    # Structured LLM
    #          ↓
    # ReviewAnalysis
    result = chain.invoke({
        "review": review
    })

    # Display the structured result returned by the LLM.
    st.write(result)