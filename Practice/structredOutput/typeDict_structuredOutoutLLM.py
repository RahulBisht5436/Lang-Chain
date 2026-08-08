import os
import sys
from pathlib import Path

from typeDict_structuredOutout import ReviewAnalysis

import streamlit as st
from langchain_core.prompts import PromptTemplate

# Get the absolute path of the parent directory
current_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(current_dir))

from llm.openAI_llm import llm

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

structured_llm = llm.with_structured_output(ReviewAnalysis)
chain = template | structured_llm

st.header("Sentiment Analysis")
review = st.text_input("Enter the review")

if st.button("Analyse Review"):
    result = chain.invoke({"review": review})
    st.write(result)
