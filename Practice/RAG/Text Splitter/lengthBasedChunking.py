
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt
from langchain_text_splitters import CharacterTextSplitter


# ---------------------------------------------------------
# Add the project's root directory to Python's import path.
#
# Path(__file__)              -> Current Python file
# .resolve()                  -> Converts it to an absolute path
# .parents[2]                 -> Goes two directories up
#
# Example:
# Practice/
# ├── RAG/
# │   └── pyPDFLaoder.py
# └── <this file>
#
# parents[2] points to the Practice/ directory.
#
# This allows us to import modules like:
#     from RAG.pyPDFLaoder import PDFLoader
# ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Import our custom PDFLoader function.
# PDFLoader is responsible for loading the PDF and
# converting its content into LangChain Document objects.
from RAG.pyPDFLaoder import PDFLoader


# ---------------------------------------------------------
# Create a CharacterTextSplitter.
#
# CharacterTextSplitter divides large documents into
# smaller chunks based on the number of characters.
#
# chunk_size=50
#     Each chunk will contain approximately 50 characters.
#
# chunk_overlap=10
#     10 characters from the previous chunk are repeated
#     in the next chunk.
#
# Overlap helps preserve context between neighboring chunks.
#
# Example:
#
# Chunk 1: "LangChain is a framework for building LLM"
#                                  ↑
#                         last 10 characters
#
# Chunk 2: "framework for building LLM applications..."
#          ↑
#      overlapping text
# ---------------------------------------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)


# ---------------------------------------------------------
# Load the PDF file.
#
# "company.pdf" is passed to our custom PDFLoader.
#
# The loader reads the PDF and returns a list of
# LangChain Document objects.
#
# Each Document generally contains:
#     - page_content -> text extracted from the PDF
#     - metadata     -> information such as page number/source
# ---------------------------------------------------------
docs = PDFLoader("company.pdf")


# You can uncomment this line if you want to inspect
# the documents returned by the PDF loader.
#
# print("======================>>>>>", docs)


# ---------------------------------------------------------
# Split the loaded documents into smaller chunks.
#
# split_documents() takes the list of Document objects
# and splits their page_content into smaller Document chunks.
#
# The result is another list of Document objects.
# ---------------------------------------------------------
result = text_splitter.split_documents(docs)


# Print the generated chunks so we can see
# how the PDF text was divided.
print(result)
