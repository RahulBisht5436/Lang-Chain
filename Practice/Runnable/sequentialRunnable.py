import os
import sys
from pathlib import Path

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

current_dir = Path(__file__).resolve().parents[1]
# Add the project root directory to Python's module search path.
sys.path.insert(0, str(current_dir))



# Import the LLM configuration/object from our project
from llm.openAI_llm import llm


# Output parser
parser = StrOutputParser()


# Prompt template
promptTemplate = PromptTemplate(
    template="""
                You are a helpful and knowledgeable assistant.

                Answer the user's query clearly and accurately.

                User Query:
                {topic}

                Provide a concise and easy-to-understand answer.
            """,
    input_variables=["topic"]
)


# Get user input
question = input("Enter the Query: ")

if question:
    # Create RunnableSequence
    chain = RunnableSequence(
        promptTemplate,
        llm,
        parser
    )

    # Execute the chain
    answer = chain.invoke({"topic": question})

    print(answer)