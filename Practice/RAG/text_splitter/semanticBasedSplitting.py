import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# SemanticChunker uses embeddings to determine where
# the meaning/topic of the text changes.
from langchain_experimental.text_splitter import SemanticChunker

# OpenAIEmbeddings converts text into numerical vectors
# so that SemanticChunker can compare the meaning of
# different parts of the document.
from langchain_openai import OpenAIEmbeddings


# Load environment variables from the .env file.
#
# Your .env file should contain something like:
#
# OPENAI_API_KEY=your_api_key_here
load_dotenv()


# Add the project root directory to Python's module
# search path so that our custom RAG package can be imported.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2])
)


# Import our custom PDF loader.
from RAG.pyPDFLaoder import PDFLoader


# Load the PDF.
#
# PDFLoader returns a list of LangChain Document objects.
docs = PDFLoader("pythonCode.pdf")


# Create the embedding model.
#
# The embedding model converts sentences/text into
# numerical vectors representing their semantic meaning.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Create the SemanticChunker.
#
# Unlike CharacterTextSplitter, SemanticChunker does not
# primarily depend on a fixed character count.
#
# It compares the semantic meaning of neighboring sentences
# and tries to identify meaningful boundaries between topics.
splitter = SemanticChunker(embeddings)


# Split the Document objects into semantic chunks.
#
# The result is a list of Document objects.
#
# Each Document contains:
#
#   page_content -> semantically grouped text
#   metadata     -> original document metadata
chunks = splitter.split_documents(docs)


# Display the generated chunks.
for index, chunk in enumerate(chunks):

    print("=" * 80)
    print(f"CHUNK {index + 1}")
    print("=" * 80)

    # Display the actual text in the semantic chunk.
    print(chunk.page_content)

    # Display metadata associated with the chunk.
    print("\nMetadata:")
    print(chunk.metadata)

    print()
