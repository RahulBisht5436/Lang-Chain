import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------

# PromptTemplate is used to create a reusable prompt
# with variables such as {topic}.
from langchain_core.prompts import PromptTemplate

# RunnableLambda allows us to use a normal Python function
# as a step inside a LangChain Runnable chain.
from langchain_core.runnables import RunnableLambda

# StrOutputParser converts the LLM response into a simple
# Python string.
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------

# BeautifulSoup is used to parse HTML and extract
# meaningful text from HTML elements.
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------

# Get the directory of the current Python file and move
# up two levels to find the project root directory.
current_dir = Path(__file__).resolve().parents[1]

# Add the project root to Python's module search path.
# This allows us to import modules from our project.
sys.path.insert(0, str(current_dir))


# ---------------------------------------------------------
# Import our LLM configuration
# ---------------------------------------------------------

# Import the LLM object configured in our project.
from llm.openAI_llm import llm


# ---------------------------------------------------------
# Normal Python function
# ---------------------------------------------------------

def cleanupFunction(data):
    """
    Takes HTML as input and extracts only the readable text.

    Example:

        <h1>Hello</h1>
        <p>Welcome to Python</p>

    becomes:

        Hello Welcome to Python
    """

    # Parse the HTML string.
    soup = BeautifulSoup(data, "html.parser")

    # Extract all text from the HTML.
    #
    # " " -> separates different HTML elements with a space.
    # strip=True -> removes unnecessary whitespace.
    return soup.get_text(" ", strip=True)


# ---------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------

promptTemplate = PromptTemplate(
    template="""
You are a helpful and knowledgeable assistant.

Answer the user's query clearly and accurately.

User Query:
{topic}

Provide a concise and easy-to-understand answer.
""",

    # The prompt expects a variable called "topic".
    input_variables=["topic"]
)


# ---------------------------------------------------------
# Output Parser
# ---------------------------------------------------------

# Converts the LLM output into a normal Python string.
parser = StrOutputParser()


# ---------------------------------------------------------
# Create the LangChain Runnable chain
# ---------------------------------------------------------

chain = (
    # Step 1:
    # Pass the HTML through our normal Python function.
    #
    # RunnableLambda converts cleanupFunction into
    # a LangChain Runnable.
    RunnableLambda(cleanupFunction)

    # Step 2:
    # Put the cleaned text into the prompt.
    | promptTemplate

    # Step 3:
    # Send the formatted prompt to the LLM.
    | llm

    # Step 4:
    # Convert the LLM response into a plain string.
    | parser
)


# ---------------------------------------------------------
# Get HTML input from the user
# ---------------------------------------------------------

# sys.stdin.read() allows us to paste MULTILINE HTML.
#
# On Windows PowerShell:
# Paste the HTML
# Press Ctrl + Z
# Press Enter
print("Paste your HTML and press Ctrl+Z, then Enter:")

topic = sys.stdin.read()


# ---------------------------------------------------------
# Execute the chain
# ---------------------------------------------------------

# Check that the user actually provided some content.
if topic.strip():

    # Send the HTML into the first Runnable.
    #
    # Flow:
    #
    # HTML
    #   ↓
    # cleanupFunction
    #   ↓
    # Clean text
    #   ↓
    # PromptTemplate
    #   ↓
    # LLM
    #   ↓
    # StrOutputParser
    #   ↓
    # Final answer
    answer = chain.invoke(topic)

    # Display the final response.
    print(answer)