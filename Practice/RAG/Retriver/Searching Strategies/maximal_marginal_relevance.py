import sys
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. ADD PROJECT ROOT TO PYTHON PATH
# ============================================================
#
# Get the root directory of the Practice project.
#
# This allows Python to find custom packages such as:
#
#     RAG/
#     llm/
#
# parents[3] moves from the current file's directory
# up to the Practice/ project directory.
#

current_dir = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT EMBEDDING MODEL, CHROMA AND LLM
# ============================================================
#
# The same embedding model that was used while creating
# the Chroma vector database must be used here.
#
# The embedding model converts the user's query into a
# numerical vector so that Chroma can perform similarity
# calculations.
#

from llm.openaiEmbedding import embeddings
from langchain_chroma import Chroma

#
# Import the Large Language Model.
#
# The LLM can later use the retrieved documents as context
# to generate the final answer.
#

from llm.openAI_llm import llm


# ============================================================
# 3. CREATE OUTPUT PARSER
# ============================================================
#
# StrOutputParser converts the LLM's output into a normal
# Python string.
#

parser = StrOutputParser()


# ============================================================
# 4. CREATE PROMPT TEMPLATE
# ============================================================
#
# This prompt defines how the LLM should answer the user's
# question.
#
# {query}
#     ↓
# Will contain the user's question.
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
# Connect to the existing Chroma vector database.
#
# We are NOT creating a new vector database here.
#
# The database was previously created and persisted in:
#
#     chroma_db_persist
#
# collection_name="langchain"
#     ↓
# Specifies which Chroma collection we want to use.
#
# embedding_function=embeddings
#     ↓
# Specifies the embedding model that Chroma will use to
# convert the user's query into an embedding vector.
#

vectorstore = Chroma(
    collection_name="langchain",

    # Same embedding model used when creating the vectors.
    embedding_function=embeddings,

    # Location of the persisted Chroma database.
    persist_directory="chroma_db_persist"
)


# ============================================================
# 6. CREATE LANGCHAIN CHAIN
# ============================================================
#
# The | operator connects LangChain components together.
#
# Flow:
#
# PromptTemplate
#       ↓
#      LLM
#       ↓
# StrOutputParser
#
# NOTE:
# This chain is created here but is not being invoked in
# the current code yet.
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
#     What are the company details?
#

query = input("Ask for the query:\n")


# ============================================================
# 8. CREATE A VECTOR RETRIEVER USING MMR
# ============================================================
#
# as_retriever() converts the Chroma vector store into a
# LangChain Retriever.
#
# search_type="mmr"
# -----------------
# Tells LangChain to use Maximal Marginal Relevance (MMR)
# instead of normal similarity search.
#
# MMR tries to select documents that are:
#
#     1. Relevant to the user's query
#     2. Different from each other
#
# This helps reduce duplicate or highly similar documents.
#
# search_kwargs:
#
# k=2
# ----
# Number of final documents that the Retriever should return.
#
# lambda_mult=0.5
# ----------------
# Controls the balance between:
#
#     Relevance  ←→  Diversity
#
# A value closer to 1:
#     More emphasis on relevance.
#
# A value closer to 0:
#     More emphasis on diversity.
#
# 0.5:
#     Approximately balanced relevance and diversity.
#

if query:

    vectorInitiater = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "lambda_mult": 0.5
        }
    )


    # ========================================================
    # 9. INVOKE THE MMR RETRIEVER
    # ========================================================
    #
    # invoke(query) sends the user's query to the Retriever.
    #
    # Internally, the process is approximately:
    #
    # User Query
    #      ↓
    # Embedding Model
    #      ↓
    # Query Vector
    #      ↓
    # Chroma
    #      ↓
    # Candidate Documents
    #      ↓
    # MMR
    #      ↓
    # Relevant + Diverse Documents
    #      ↓
    # Top 2 Documents
    #
    # The returned result is a list of LangChain Document
    # objects.
    #

    result = vectorInitiater.invoke(query)


    # ========================================================
    # 10. DISPLAY RETRIEVED DOCUMENTS
    # ========================================================
    #
    # Each item in result is a Document object.
    #
    # doc.page_content
    #     → Contains the actual text retrieved from Chroma.
    #
    # doc.metadata
    #     → Contains additional information about the document,
    #       such as source, file name, page number, etc.
    #
    # enumerate(..., start=1)
    #     → Numbers the retrieved documents starting from 1.
    #

    for i, doc in enumerate(result, start=1):

        print(f"\n{'=' * 60}")
        print(f"Document {i}")
        print(f"{'=' * 60}")

        # Print the actual retrieved document content.
        print(f"Content:\n{doc.page_content}")

        # Print metadata associated with the document.
        print(f"\nMetadata:\n{doc.metadata}")