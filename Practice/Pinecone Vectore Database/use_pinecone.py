import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI


# =========================================================
# Load Environment Variables
# =========================================================

# Load variables from .env
load_dotenv()


# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Make sure Pinecone API key exists
if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set in .env"
    )


# Make sure OpenAI API key exists
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set in .env"
    )


# =========================================================
# Pinecone Configuration
# =========================================================

INDEX_NAME = "company-knowledge"

NAMESPACE = "company-documents"


# Number of documents/chunks we want Pinecone
# to return for every user question.
TOP_K = 5


# =========================================================
# Initialize Pinecone
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# Get our existing Pinecone index
index = pc.Index(INDEX_NAME)


# =========================================================
# Initialize LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)


# =========================================================
# Search Pinecone
# =========================================================

def search_pinecone(question):

    """
    Search Pinecone using the user's question.

    Since our Pinecone index uses integrated embedding,
    Pinecone automatically converts the question into
    an embedding before performing semantic search.
    """

    print("\nSearching Pinecone...")
    print(f"Question: {question}")


    # Perform semantic search.
    #
    # inputs={"text": question}
    #
    # Pinecone will:
    #
    # Question
    #    ↓
    # Embedding
    #    ↓
    # Vector similarity search
    #    ↓
    # Most relevant chunks
    #
    results = index.search(
        namespace=NAMESPACE,

        # Number of chunks to retrieve
        top_k=TOP_K,

        # User's natural-language question.
        #
        # Pinecone's integrated embedding model will
        # convert this text into an embedding.
        inputs={
            "text": question
        },

        # Only return the fields we actually need.
        fields=[
            "chunk_text",
            "page",
            "source"
        ]
    )


    return results


# =========================================================
# Extract Retrieved Context
# =========================================================

def get_context(results):

    """
    Convert Pinecone search results into a single
    context string that can be sent to the LLM.
    """

    context_parts = []


    # Pinecone returns matching records as hits.
    for hit in results.result.hits:

        # Get the actual text stored in Pinecone.
        chunk_text = hit.fields.get(
            "chunk_text",
            ""
        )


        # Get metadata
        page = hit.fields.get(
            "page",
            "Unknown"
        )

        source = hit.fields.get(
            "source",
            "Unknown"
        )


        # Add this chunk to the context.
        context_parts.append(
            f"""
Source: {source}
Page: {page}

{chunk_text}
"""
        )


    # Join all retrieved chunks together.
    return "\n\n-----------------------------\n\n".join(
        context_parts
    )


# =========================================================
# Ask LLM
# =========================================================

def ask_llm(question, context):

    """
    Send the user's question + retrieved Pinecone
    context to the LLM.
    """

    prompt = f"""
You are a helpful AI assistant answering questions
about a company.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say:

"I could not find this information in the company documents."

Do not make up information.

-----------------------------
CONTEXT
-----------------------------

{context}

-----------------------------
USER QUESTION
-----------------------------

{question}

-----------------------------
ANSWER
-----------------------------
"""


    # Send prompt to the LLM.
    response = llm.invoke(prompt)


    # ChatOpenAI returns an AIMessage.
    #
    # .content contains the actual answer.
    return response.content


# =========================================================
# Complete RAG Pipeline
# =========================================================

def ask_question(question):

    """
    Complete RAG pipeline:

    User Question
          ↓
    Pinecone Search
          ↓
    Relevant Chunks
          ↓
    Build Context
          ↓
    Send Context + Question to LLM
          ↓
    Final Answer
    """


    # -----------------------------------------------------
    # Step 1: Search Pinecone
    # -----------------------------------------------------

    results = search_pinecone(question)


    # -----------------------------------------------------
    # Step 2: Extract relevant chunks
    # -----------------------------------------------------

    context = get_context(results)


    # -----------------------------------------------------
    # Step 3: Send context + question to LLM
    # -----------------------------------------------------

    answer = ask_llm(
        question=question,
        context=context
    )


    return answer


# =========================================================
# Main Program
# =========================================================

if __name__ == "__main__":

    print("\n======================================")
    print("     COMPANY RAG SYSTEM")
    print("======================================\n")


    # Ask the user for a question.
    question = input(
        "Ask a question about the company: "
    )


    # Run the complete RAG pipeline.
    answer = ask_question(question)


    # Display the final answer.
    print("\n======================================")
    print("ANSWER")
    print("======================================\n")

    print(answer)