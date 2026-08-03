# ==========================================================
# Import Required Libraries
# ==========================================================

# Embedding model that converts text into vectors.
from llm.ollamaEmbedding import embeddings

# Reads text files and converts them into LangChain Documents.
from langchain_community.document_loaders import TextLoader

# Splits large documents into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma Vector Database
from langchain_chroma import Chroma

# Used for creating platform-independent file paths.
from pathlib import Path


# ==========================================================
# STEP 1 : Locate the text file
# ==========================================================

# __file__ refers to the current Python file.
# .parent returns the directory containing this file.
current_dir = Path(__file__).parent

# Build the path to joblisting.txt
file_path = current_dir / "data" / "joblisting.txt"


# ==========================================================
# STEP 2 : Load the Document
# ==========================================================

# TextLoader converts the text file into a list of
# LangChain Document objects.
documents = TextLoader(str(file_path)).load()

print("=" * 80)
print("DOCUMENT LOADED")
print("=" * 80)

for document in documents:
    print(document)

print()


# ==========================================================
# STEP 3 : Create the Text Splitter
# ==========================================================

# Large documents are split into smaller chunks so that
# embeddings can be generated efficiently.
#
# chunk_size    -> Maximum characters per chunk.
# chunk_overlap -> Shared characters between chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)


# ==========================================================
# STEP 4 : Split the Documents
# ==========================================================

chunks = text_splitter.split_documents(documents)

print("=" * 80)
print(f"TOTAL CHUNKS CREATED : {len(chunks)}")
print("=" * 80)

for index, chunk in enumerate(chunks, start=1):

    print(f"\nChunk {index}")
    print("-" * 60)
    print(chunk.page_content)

print()


# ==========================================================
# STEP 5 : Create the Vector Database
# ==========================================================

# Chroma performs two operations:
#
# 1. Generate embeddings for every chunk.
# 2. Store those embeddings in the vector database.
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("=" * 80)
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 80)
print()


# ==========================================================
# STEP 6 : Create a Retriever
# ==========================================================

# A Retriever is a wrapper around the Vector Store.
#
# Instead of manually:
#
# Query
#   ↓
# Generate Embedding
#   ↓
# Search Vector Database
#   ↓
# Return Documents
#
# the Retriever performs all these steps internally.
retriever = db.as_retriever()


# ==========================================================
# STEP 7 : Take User Input
# ==========================================================

query = input("What are you searching for? : ")


# ==========================================================
# STEP 8 : Retrieve Similar Documents
# ==========================================================

# invoke() automatically:
#
# 1. Converts the query into an embedding.
# 2. Searches the vector database.
# 3. Returns the most similar Document objects.
results = retriever.invoke(query)

print(f"\nRetriever returned {len(results)} matching documents.\n")


# ==========================================================
# STEP 9 : Display Search Results
# ==========================================================

print("=" * 80)
print("SEARCH RESULTS")
print("=" * 80)

for index, document in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("-" * 80)

    # Unique identifier assigned by Chroma
    print("Document ID:")
    print(document.id)

    # Metadata such as source file
    print("\nMetadata:")
    print(document.metadata)

    # Actual retrieved text
    print("\nPage Content:")
    print(document.page_content)

    print("-" * 80)


# ==========================================================
# END OF PROGRAM
# ==========================================================