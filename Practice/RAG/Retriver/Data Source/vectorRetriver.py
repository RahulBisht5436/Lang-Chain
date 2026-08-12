import sys
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. ADD PROJECT ROOT TO PYTHON PATH
# ============================================================
#
# Get the Practice/ project root directory.
#
# __file__ -> current Python file
# parents[2] -> Practice/
#
# Adding the project root to sys.path allows Python to
# find our custom packages such as:
#
#     RAG/
#     llm/
#
current_dir = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT EMBEDDING MODEL AND LLM
# ============================================================
#
# IMPORTANT:
# We must use the SAME embedding model that was used when
# creating the Chroma vector database.
#
# The embedding model converts text into numerical vectors.
#
# Example:
#
# "What is Python?"
#       ↓
# Embedding Model
#       ↓
# [0.12, -0.45, 0.78, ...]
#
from llm.openaiEmbedding import embeddings
from langchain_chroma import Chroma


# Import the Large Language Model (LLM).
#
# The LLM will receive:
#     1. The user's question
#     2. Relevant documents retrieved from Chroma
#
# and generate the final natural-language answer.
from llm.openAI_llm import llm


# ============================================================
# 3. CREATE OUTPUT PARSER
# ============================================================
#
# StrOutputParser converts the LLM's output into a simple
# Python string.
#
# Without the parser, the LLM response may be returned
# as a more complex response object.
#
parser = StrOutputParser()


# ============================================================
# 4. CREATE PROMPT TEMPLATE
# ============================================================
#
# This prompt tells the LLM how to use the retrieved
# information to answer the user's question.
#
# {embedding_variable}
#     ↓
# Contains the documents retrieved from Chroma.
#
# {query}
#     ↓
# Contains the user's original question.
#
prompt = PromptTemplate(
    template="""
You are an assistant that answers questions using the
provided context.

Question:
{query}

Provide an appropriate answer based on the provided context.
""",
    input_variables=["query"]
)


# ============================================================
# 5. LOAD EXISTING CHROMA VECTOR DATABASE
# ============================================================
#
# We are NOT creating the vector database here.
#
# The vector database was already created in another file
# using:
#
#     Chroma.from_documents(...)
#
# Here, we simply connect to the existing persistent
# Chroma database.
#
vectorstore = Chroma(
    collection_name="langchain",

    # The same embedding model is required to convert
    # the user's query into a vector for similarity search.
    embedding_function=embeddings,

    # Location where the Chroma database is persisted.
    persist_directory="chroma_db_persist"
)


# ============================================================
# 6. CREATE THE LANGCHAIN CHAIN
# ============================================================
#
# The | operator connects multiple LangChain components.
#
# Flow:
#
# PromptTemplate
#       ↓
#      LLM
#       ↓
# StrOutputParser
#
# First, the prompt is created.
# Then the prompt is sent to the LLM.
# Finally, the LLM response is converted into a string.
#
chainOutput = prompt | llm | parser


# ============================================================
# 7. GET USER QUERY
# ============================================================
#
# Ask the user to enter a question.
#
# Example:
#
#     What technologies does the company use?
#
query = input("Ask for the query:\n")


# ============================================================
# 8. PERFORM SIMILARITY SEARCH
# ============================================================
#
# Only continue if the user entered a query.
#
if query:

    # Search the Chroma vector database for documents
    # that are semantically similar to the user's question.
    #
    # The query is converted into an embedding vector
    # using the embedding model.
    #
    # Chroma then compares the query vector with the
    # stored document vectors and returns the most
    # relevant documents.
    #
    # Example:
    #
    # User Query
    #      ↓
    # Embedding Model
    #      ↓
    # Query Vector
    #      ↓
    # Chroma Similarity Search
    #      ↓
    # Relevant Documents
    # embedding_variable = vectorstore.similarity_search(query)
    vectorInitiater = vectorstore.as_retriever(
        search_kwargs={"k":2}
    )
    result = vectorInitiater.invoke(query)

    
    #
    for i, doc in enumerate(result, start=1):
        print(f"\n{'=' * 60}")
        print(f"Document {i}")
        print(f"{'=' * 60}")
        print(f"Content:\n{doc.page_content}")
        print(f"\nMetadata:\n{doc.metadata}")