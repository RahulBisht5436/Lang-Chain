import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt

# RecursiveCharacterTextSplitter is used to divide large text into
# smaller chunks.
#
# Language is imported so that we can tell the splitter that the
# document contains Python code.
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)


# Add the project's root directory to Python's module search path.
#
# __file__ -> current Python file
# parents[2] -> moves two directories upward
#
# This allows Python to find our custom RAG package.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2])
)


# Import our custom PDF loader.
#
# PDFLoader() loads the PDF and returns a list of LangChain
# Document objects.
#
# Each Document generally contains:
#
#   doc.page_content -> text extracted from the PDF
#   doc.metadata     -> information about the source/page
from RAG.pyPDFLaoder import PDFLoader


# Load the Python code PDF.
#
# The PDF loader extracts the text from the PDF and returns
# a list of Document objects.
#
# Example:
#
# [
#     Document(
#         page_content="class User: ...",
#         metadata={"page": 0, "source": "pythonCode.pdf"}
#     ),
#     Document(
#         page_content="def login(): ...",
#         metadata={"page": 1, "source": "pythonCode.pdf"}
#     )
# ]
docs = PDFLoader("pythonCode.pdf")


# Create a RecursiveCharacterTextSplitter specifically
# configured for Python code.
#
# from_language() tells LangChain that the content has
# Python syntax and structure.
#
# Therefore, the splitter can prefer Python-related
# boundaries instead of blindly splitting characters.
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,

    # Maximum approximate size of each chunk.
    #
    # 100 is intentionally small here so that you can
    # easily observe how the code is being split.
    #
    # For a real RAG application, you would normally
    # experiment with a larger value such as 500, 1000,
    # etc.
    chunk_size=300,

    # Number of characters that can be repeated between
    # consecutive chunks.
    #
    # This helps preserve some context when a logical
    # piece of code crosses a chunk boundary.
    chunk_overlap=50
)


# Split the Document objects into smaller Document objects.
#
# Input:
#     List[Document]
#
# Output:
#     List[Document]
#
# IMPORTANT:
# split_documents() keeps the Document structure and
# metadata instead of returning only plain strings.
splitted_docs = splitter.split_documents(docs)


# Iterate through every newly created chunk.
for index, doc in enumerate(splitted_docs):

    # Print a separator so that individual chunks are
    # easy to identify in the terminal.
    print("=" * 60)

    # Display the chunk number.
    print(f"CHUNK {index + 1}")

    print("=" * 60)

    # Print the actual Python code contained in this chunk.
    #
    # page_content contains the text extracted from the
    # original PDF and then divided by the splitter.
    print(doc.page_content)

    # Print the metadata associated with this chunk.
    #
    # Metadata can contain information such as:
    #
    # {
    #     "source": "pythonCode.pdf",
    #     "page": 0
    # }
    #
    # Keeping metadata is very useful in RAG because later
    # you can identify where a retrieved piece of information
    # originally came from.
    print("\nMetadata:")
    print(doc.metadata)

    # Print an empty line to make the terminal output easier
    # to read before displaying the next chunk.
    print()
