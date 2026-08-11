import sys
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. ADD PROJECT ROOT TO PYTHON PATH
# ============================================================
#
# Get the Practice/ project root directory.
#
# __file__   -> path of the current Python file
# parents[2] -> Practice/ project directory
#
# Adding the project root to sys.path allows Python to find
# our custom packages such as:
#
#     RAG/
#     llm/
#
current_dir = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT EMBEDDING MODEL AND LLM
# ============================================================
#
# IMPORTANT:
#
# We must use the SAME embedding model that was used when
# creating the FAISS vector store.
#
# The embedding model converts text into numerical vectors.
#
# Example:
#
#     "What is Python?"
#             ↓
#     Embedding Model
#             ↓
#     [0.12, -0.45, 0.78, ...]
#
from llm.openaiEmbedding import embeddings


# Import the Large Language Model (LLM).
#
# The LLM will receive:
#
# 1. The user's question
# 2. The relevant documents retrieved from FAISS
#
# and will generate the final natural-language answer.
from llm.openAI_llm import llm


# ============================================================
# 3. CREATE OUTPUT PARSER
# ============================================================
#
# StrOutputParser converts the LLM's response into a
# normal Python string.
#
# Flow:
#
#     LLM Response
#          ↓
#     StrOutputParser
#          ↓
#     Python String
#
parser = StrOutputParser()


# ============================================================
# 4. CREATE PROMPT TEMPLATE
# ============================================================
#
# The prompt tells the LLM to answer the user's question
# using the context retrieved from the FAISS vector store.
#
# {context}
#     ↓
# Contains the relevant Document objects retrieved
# from FAISS.
#
# {query}
#     ↓
# Contains the original question asked by the user.
#
prompt = PromptTemplate(
    template="""
You are an assistant that answers questions using the
provided context.

Context:
{context}

Question:
{query}

Provide an appropriate answer based on the provided context.
""",
    input_variables=["query", "context"]
)


# ============================================================
# 5. LOAD EXISTING FAISS VECTOR STORE
# ============================================================
#
# We are NOT creating the FAISS vector store here.
#
# The vector store was already created and saved in another
# file using:
#
#     FAISS.from_documents(...)
#     vectorstore.save_local("faiss_db")
#
# Here, we load the previously saved FAISS index from disk.
#
# The "faiss_db" directory should contain files such as:
#
#     faiss_db/
#     ├── index.faiss
#     └── index.pkl
#
# IMPORTANT:
#
# The same embedding model must be provided because it will
# be used to convert the user's query into a vector during
# similarity search.
#
vectorstore = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")


# ============================================================
# 6. CREATE THE LANGCHAIN CHAIN
# ============================================================
#
# The "|" operator connects LangChain components together.
#
# Flow:
#
#     PromptTemplate
#          ↓
#         LLM
#          ↓
#     StrOutputParser
#          ↓
#      Final String
#
# The prompt is first populated with the user's query and
# retrieved context.
#
# Then it is sent to the LLM.
#
# Finally, StrOutputParser converts the LLM response into
# a normal string.
#
chain_output = prompt | llm | parser


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
# Continue only if the user entered a query.
#
if query:

    # --------------------------------------------------------
    # Convert the user's query into an embedding vector
    # and search the FAISS index for similar vectors.
    #
    # FAISS compares the query vector against the vectors
    # stored in the vector index.
    #
    # The most relevant Document objects are returned.
    #
    # Flow:
    #
    # User Query
    #      ↓
    # Embedding Model
    #      ↓
    # Query Vector
    #      ↓
    # FAISS Similarity Search
    #      ↓
    # Relevant Documents
    #
    # IMPORTANT:
    #
    # similarity_search() returns Document objects,
    # NOT the actual embedding vectors.
    #
    retrieved_documents = vectorstore.similarity_search(
        query,
        k=3
    )


    # ========================================================
    # 9. SEND QUERY + RETRIEVED DOCUMENTS TO THE LLM
    # ========================================================
    #
    # Send two pieces of information to the prompt:
    #
    # query:
    #     The original question asked by the user.
    #
    # context:
    #     The relevant documents retrieved from FAISS.
    #
    # The PromptTemplate combines these values and creates
    # the final prompt that will be sent to the LLM.
    #
    result = chain_output.invoke({
        "query": query,
        "context": retrieved_documents
    })


    # ========================================================
    # 10. DISPLAY FINAL ANSWER
    # ========================================================
    #
    # StrOutputParser has converted the LLM response into
    # a normal Python string.
    #
    print(result)