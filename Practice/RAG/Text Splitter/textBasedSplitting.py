import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Add the project root directory to Python's module search path.
# This allows us to import our custom PDFLoader module from the RAG package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Import our custom PDFLoader function.
# PDFLoader loads the PDF and returns a list of LangChain Document objects.
from RAG.pyPDFLaoder import PDFLoader


# Load the company.pdf file.
#
# The loader does NOT return plain strings.
# It returns a list of Document objects.
#
# Each Document generally contains:
#   - page_content -> actual text extracted from the PDF
#   - metadata     -> information such as page number and source
docs = PDFLoader("company.pdf")


# Create a RecursiveCharacterTextSplitter.
#
# chunk_size:
#   Maximum size we want for each text chunk.
#
# chunk_overlap:
#   Number of characters that can be repeated between
#   consecutive chunks.
#
# Example:
#
#   Chunk 1: "Python is a programming language..."
#   Chunk 2: "...programming language used for..."
#
# The repeated part provides some context between chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)


# Split the Document objects into smaller Document objects.
#
# split_documents() takes:
#
#     List[Document]
#
# and returns:
#
#     List[Document]
#
# The important thing is that the metadata is preserved.
# Therefore, we still know which page/source the chunk came from.
splitted_docs = splitter.split_documents(docs)


# Iterate through every newly created chunk.
for doc in splitted_docs:

    # Print the actual text contained inside the chunk.
    print(doc.page_content)

    # Print metadata associated with the chunk.
    # For example:
    # {'source': 'company.pdf', 'page': 0}
    print(doc.metadata)

    # Print a separator to make the output easier to read.
    print("-" * 50)
