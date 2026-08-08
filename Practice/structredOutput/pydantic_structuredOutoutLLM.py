import os
import sys
from pathlib import Path

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
# This allows us to import modules from the Practice directory.

current_dir = Path(__file__).resolve().parents[1]

# Add the project root directory to Python's module search path.
sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT THE LLM
# ============================================================

# Import the LLM configuration/object from our project.
from llm.openAI_llm import llm

# Import the Pydantic schema that defines
# the structure expected from the LLM's output.
from pydantic_StructuredOutput import ReviewAnalysis


# ============================================================
# 3. CREATE THE PROMPT TEMPLATE
# ============================================================

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

    input_variables=["review"],
)


# ============================================================
# 4. CONVERT NORMAL LLM INTO STRUCTURED-OUTPUT LLM
# ============================================================

# ReviewAnalysis is a Pydantic BaseModel.
#
# with_structured_output() tells LangChain to generate
# the response according to the Pydantic schema.
#
# The returned result will be a Pydantic object.

structured_llm = llm.with_structured_output(ReviewAnalysis)


# ============================================================
# 5. CREATE THE LANGCHAIN CHAIN
# ============================================================

# Flow:
#
# User Review
#      ↓
# PromptTemplate
#      ↓
# Structured LLM
#      ↓
# ReviewAnalysis (Pydantic object)

chain = template | structured_llm


# ============================================================
# 6. STREAMLIT USER INTERFACE
# ============================================================

st.header("Sentiment Analysis")


# Create a text input box where the user can enter
# a product review.

review = st.text_input("Enter the review")


# Create an "Analyse Review" button.

if st.button("Analyse Review"):

    # Send the user's review into the LangChain chain.
    result = chain.invoke({
        "review": review
    })

    # Display the Pydantic object.
    st.write(result)

    # You can also convert the Pydantic object
    # into a normal Python dictionary.
    st.write(result.model_dump())
