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
# __file__
#     ↓
# Represents the current Python file.
#
# resolve()
#     ↓
# Converts the file path into an absolute path.
#
# parents[3]
#     ↓
# Moves three directories upward to reach the Practice/
# project root.
#
# Adding the project root to sys.path allows Python to find
# our custom packages such as:
#
#     RAG/
#     llm/
#
# Example project structure:
#
# Practice/
# ├── RAG/
# ├── llm/
# └── ...
#

current_dir = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT EMBEDDING MODEL, CHROMA AND LLM
# ============================================================
#
# Import the same embedding model that was used while
# creating the Chroma vector database.
#
# IMPORTANT:
#
# The embedding model used during document indexing and the
# embedding model used during retrieval should be compatible.
#
# The embedding model converts text into a numerical vector.
#
# Example:
#
# "What are company details?"
#             ↓
#      Embedding Model
#             ↓
# [0.12, -0.45, 0.78, ...]
#
# This vector is then used by Chroma to perform vector search.
#

from llm.openaiEmbedding import embeddings

from langchain_chroma import Chroma


# Import the Large Language Model.
#
# In this implementation, the LLM has an important additional
# responsibility:
#
# MultiQueryRetriever uses this LLM to generate multiple
# alternative versions of the user's query.
#
# Example:
#
# Original:
# "What are company revenue sources?"
#
# LLM may generate:
#
# 1. "How does the company generate income?"
# 2. "What are the company's revenue streams?"
# 3. "What products generate company revenue?"
#

from llm.openAI_llm import llm


# Import LangChain's MultiQueryRetriever.
#
# MultiQueryRetriever takes:
#
#     User Query
#          ↓
#         LLM
#          ↓
# Multiple Query Variations
#          ↓
# Existing Retriever
#          ↓
# Combined Documents
#
# It does not replace Chroma.
#
# Instead, it works ON TOP of an existing Retriever.
#

from langchain_classic.retrievers import MultiQueryRetriever


# ============================================================
# 3. CREATE OUTPUT PARSER
# ============================================================
#
# StrOutputParser converts an LLM response into a normal
# Python string.
#
# For example, an LLM response may be represented by a
# LangChain response object.
#
# StrOutputParser extracts the text output from that response.
#
# NOTE:
#
# The parser is used for the answer-generation chain below.
# MultiQueryRetriever internally handles the query-generation
# output separately.
#

parser = StrOutputParser()


# ============================================================
# 4. CREATE PROMPT TEMPLATE
# ============================================================
#
# This PromptTemplate defines the instructions that will be
# given to the LLM when generating the final answer.
#
# {query}
#     ↓
# This placeholder will contain the user's original question.
#
# NOTE:
#
# This prompt is currently NOT being used during retrieval.
#
# The MultiQueryRetriever has its own internal mechanism for
# asking the LLM to generate alternative queries.
#
# This prompt is intended for the later RAG answer-generation
# stage.
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
# Connect to the existing persistent Chroma vector database.
#
# We are NOT creating a new vector database here.
#
# The vector database was previously created and persisted
# in:
#
#     chroma_db_persist
#
#
# collection_name="langchain"
# ---------------------------
# Specifies the Chroma collection that we want to access.
#
#
# embedding_function=embeddings
# -----------------------------
# Specifies the embedding model that Chroma will use when
# converting the user's query into an embedding vector.
#
#
# persist_directory="chroma_db_persist"
# -------------------------------------
# Specifies where the persisted Chroma database is located.
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
# This chain is intended for generating the final answer
# from the user's question and retrieved context.
#
# NOTE:
#
# In the current code, this chain is created but NOT invoked.
#
# Your current code is only demonstrating retrieval.
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
# The input() function returns the user's input as a STRING.
#

query = input("Ask for the query:\n")


# ============================================================
# 8. CREATE A VECTOR RETRIEVER USING MMR
# ============================================================
#
# First, we convert the Chroma Vector Store into a
# LangChain Retriever.
#
# as_retriever()
#     ↓
# Creates a Retriever object.
#
# The Retriever provides a standard interface for retrieving
# relevant documents.
#
#
# search_type="mmr"
# -----------------
# Tells the Retriever to use:
#
#     Maximal Marginal Relevance (MMR)
#
# instead of normal similarity search.
#
#
# MMR tries to select documents that are:
#
#     1. Relevant to the query
#     2. Different from each other
#
# This helps reduce redundant documents.
#
#
# search_kwargs
# -------------
# Contains configuration for the retrieval operation.
#
#
# k=2
# ---
# Specifies the number of final documents that MMR should
# return for EACH generated query.
#
#
# lambda_mult=0.5
# ---------------
# Controls the balance between:
#
#     Relevance  <---------->  Diversity
#
# lambda closer to 1:
#     More importance to relevance.
#
# lambda closer to 0:
#     More importance to diversity.
#
# lambda = 0.5:
#     Approximately balanced.
#
#
# IMPORTANT:
#
# This Retriever is going to become the BASE retriever for
# MultiQueryRetriever.
#
# So the architecture is:
#
# MultiQueryRetriever
#         ↓
#    MMR Retriever
#         ↓
#      Chroma
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
    # 9. CREATE MULTI-QUERY RETRIEVER
    # ========================================================
    #
    # Now we wrap the MMR Retriever inside a
    # MultiQueryRetriever.
    #
    # MultiQueryRetriever uses the LLM to generate multiple
    # alternative versions of the user's original query.
    #
    #
    # Example:
    #
    # Original Query:
    # "What are the company revenue sources?"
    #
    # LLM may generate:
    #
    # Query 1:
    # "How does the company generate income?"
    #
    # Query 2:
    # "What are the company's revenue streams?"
    #
    # Query 3:
    # "What products or services generate revenue?"
    #
    #
    # Each generated query is then sent to the BASE retriever.
    #
    # In our case, the base retriever is the MMR Retriever.
    #
    #
    # Therefore:
    #
    #             Original Query
    #                    ↓
    #                   LLM
    #                    ↓
    #          Multiple Query Variations
    #             ↙       ↓       ↘
    #         Query 1   Query 2   Query 3
    #             ↓       ↓       ↓
    #            MMR     MMR      MMR
    #             ↓       ↓       ↓
    #          Chroma  Chroma   Chroma
    #             ↓       ↓       ↓
    #          Documents Documents Documents
    #             ↘       ↓       ↙
    #              Combined Results
    #                    ↓
    #              Unique Documents
    #

    multiQueryRetriverInitiater = MultiQueryRetriever.from_llm(
        retriever=vectorInitiater,
        llm=llm,

        # Include the original user query along with the
        # queries generated by the LLM.
        #
        # True means:
        #
        # Original Query
        #       +
        # Generated Queries
        #
        include_original=True
    )


    # ========================================================
    # 10. INVOKE THE MULTI-QUERY RETRIEVER
    # ========================================================
    #
    # invoke(query) starts the complete retrieval process.
    #
    # IMPORTANT:
    #
    # We are invoking MultiQueryRetriever, NOT the MMR
    # Retriever directly.
    #
    #
    # Complete flow:
    #
    # User Query
    #      ↓
    # MultiQueryRetriever
    #      ↓
    # LLM generates multiple queries
    #      ↓
    # ┌──────────────┬──────────────┬──────────────┐
    # ↓              ↓              ↓
    # Query 1       Query 2       Query 3
    # ↓              ↓              ↓
    # MMR            MMR            MMR
    # ↓              ↓              ↓
    # Chroma         Chroma         Chroma
    # ↓              ↓              ↓
    # Documents      Documents      Documents
    # └──────────────┴──────────────┴──────────────┘
    #                       ↓
    #              Combine Results
    #                       ↓
    #               Remove Duplicates
    #                       ↓
    #               Final Documents
    #
    # The result is a list of LangChain Document objects.
    #

    result = multiQueryRetriverInitiater.invoke(query)


    # ========================================================
    # 11. DISPLAY RETRIEVED DOCUMENTS
    # ========================================================
    #
    # The result contains the documents retrieved from Chroma.
    #
    # Each item is a LangChain Document object.
    #
    #
    # doc.page_content
    # ----------------
    # Contains the actual text/content of the retrieved
    # document.
    #
    #
    # doc.metadata
    # ------------
    # Contains additional information about the document.
    #
    # Examples:
    #
    #     source
    #     file name
    #     page number
    #     document ID
    #
    #
    # enumerate(..., start=1)
    # -----------------------
    # Gives each document a sequential number starting from 1.
    #

    for i, doc in enumerate(result, start=1):

        print(f"\n{'=' * 60}")
        print(f"Document {i}")
        print(f"{'=' * 60}")

        # Print the actual text/content retrieved from Chroma.
        print(f"Content:\n{doc.page_content}")

        # Print metadata associated with the document.
        print(f"\nMetadata:\n{doc.metadata}")