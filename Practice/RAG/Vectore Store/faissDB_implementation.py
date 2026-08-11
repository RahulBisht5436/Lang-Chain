import sys
from pathlib import Path

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
# Adding the project root to sys.path allows Python to
# find our custom local packages such as:
#
#     RAG/
#     llm/
#
current_dir = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT TEXT SPLITTER AND EMBEDDING MODEL
# ============================================================
#
# Import our custom text splitting function.
#
# textSpillterCustom():
#
# 1. Loads the PDF.
# 2. Extracts the text.
# 3. Splits the extracted text into smaller chunks.
# 4. Returns a list of LangChain Document objects.
#
from RAG.text_splitter.textBasedSplitting import textSpillterCustom


# Import the embedding model.
#
# The embedding model converts each text chunk into
# a numerical vector.
#
# Example:
#
#     Text Chunk
#         ↓
#     Embedding Model
#         ↓
#     [0.12, -0.45, 0.78, ...]
#
# The same embedding model should also be used later
# when searching the FAISS vector store.
from llm.openaiEmbedding import embeddings


# ============================================================
# 3. LOAD AND SPLIT THE PDF
# ============================================================
#
# Load company.pdf and split its content into smaller
# Document chunks.
#
# The returned value is a list of LangChain Document objects.
#
# Example:
#
# result = [
#     Document(...),
#     Document(...),
#     Document(...)
# ]
#
result = textSpillterCustom("company.pdf")


# ============================================================
# 4. CREATE FAISS VECTOR STORE
# ============================================================
#
# FAISS.from_documents() performs several operations:
#
# 1. Takes the Document chunks.
#
# 2. Sends each Document's text to the embedding model.
#
# 3. Converts each text chunk into a numerical vector.
#
# 4. Stores those vectors inside a FAISS similarity index.
#
# 5. Keeps the relationship between the vectors and
#    their corresponding Documents.
#
# Flow:
#
#     Document Chunks
#           ↓
#     Embedding Model
#           ↓
#     Numerical Vectors
#           ↓
#     FAISS Index
#
vectorstore = FAISS.from_documents(
    documents=result,
    embedding=embeddings,
)


# ============================================================
# 5. SAVE FAISS VECTOR STORE TO DISK
# ============================================================
#
# FAISS keeps the vector index in memory by default.
#
# save_local() saves the FAISS vector store to disk so
# that we can load and use it later without processing
# the PDF and creating embeddings again.
#
# This creates a directory similar to:
#
#     faiss_db/
#     ├── index.faiss
#     └── index.pkl
#
# index.faiss:
#     Contains the FAISS vector index used for
#     similarity searching.
#
# index.pkl:
#     Stores the LangChain document store and
#     mappings/metadata needed to associate search
#     results with the original Documents.
#
vectorstore.save_local("faiss_db")


# ============================================================
# 6. CONFIRM SUCCESSFUL CREATION
# ============================================================
#
# This message confirms that the FAISS vector store
# was successfully created and saved to disk.
#
print("FAISS vector store created and saved successfully!")