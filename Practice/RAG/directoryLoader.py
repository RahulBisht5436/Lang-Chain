# Import the required document loaders from LangChain
#
# DirectoryLoader  -> Loads multiple files from a directory
# TextLoader      -> Loads .txt files
# PyPDFLoader     -> Loads PDF files
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader
)

# Path is used to work with file and directory paths
# in a clean and platform-independent way
from pathlib import Path


# =========================================================
# 1. GET THE PROJECT DIRECTORY
# =========================================================

# __file__ represents the current Python file.
#
# Path(__file__).resolve()
#     -> Gets the absolute path of the current Python file.
#
# .parents[1]
#     -> Moves two levels up in the directory structure.
#
# Example:
# Practice/RAG/textLoader.py
#             ↑
#             parents[0] = RAG
#             parents[1] = Practice
current_dir = Path(__file__).resolve().parents[1]


# =========================================================
# 2. DEFINE THE DATA DIRECTORY
# =========================================================

# Build the path to the Data directory.
#
# Expected structure:
#
# Practice/
# └── RAG/
#     ├── textLoader.py
#     └── Data/
#         ├── company.txt
#         ├── employees.txt
#         ├── company.pdf
#         └── policies.pdf
#
directory_path = current_dir / "RAG" / "Data"


# =========================================================
# 3. CREATE LOADER FOR TEXT FILES
# =========================================================

# DirectoryLoader will search the Data directory
# for files matching the specified pattern.
#
# glob="*.txt"
#     -> Selects only .txt files.
#
# loader_cls=TextLoader
#     -> Tells DirectoryLoader to use TextLoader
#        to read those .txt files.
#
# Example:
#
# company.txt    -> TextLoader
# employees.txt  -> TextLoader
#
text_loader = DirectoryLoader(
    path=str(directory_path),
    glob="*.txt",
    loader_cls=TextLoader
)
# if we dont want to load every thing at a time we can use lazy_load()

# =========================================================
# 4. CREATE LOADER FOR PDF FILES
# =========================================================

# Another DirectoryLoader is required because
# PDF files need a different loader.
#
# glob="*.pdf"
#     -> Selects only .pdf files.
#
# loader_cls=PyPDFLoader
#     -> Tells DirectoryLoader to use PyPDFLoader
#        for the PDF files.
#
# Example:
#
# company.pdf   -> PyPDFLoader
# policies.pdf  -> PyPDFLoader
#
pdf_loader = DirectoryLoader(
    path=str(directory_path),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)


# =========================================================
# 5. LOAD THE TEXT DOCUMENTS
# =========================================================

# .load() searches the directory for .txt files
# and converts them into LangChain Document objects.
#
# Example result:
#
# text_docs = [
#     Document(...company.txt...),
#     Document(...employees.txt...)
# ]
#
text_docs = text_loader.load()


# =========================================================
# 6. LOAD THE PDF DOCUMENTS
# =========================================================

# .load() searches the directory for .pdf files
# and uses PyPDFLoader to extract their content.
#
# A PDF can produce multiple Document objects,
# usually one Document per page.
#
# Example:
#
# company.pdf
#     ↓
# Document(page=0)
# Document(page=1)
#
pdf_docs = pdf_loader.load()


# =========================================================
# 7. COMBINE ALL DOCUMENTS
# =========================================================

# Both .load() methods return Python lists.
#
# We can combine the two lists using +.
#
# text_docs + pdf_docs
#
# Result:
#
# docs = [
#     Document(...company.txt...),
#     Document(...employees.txt...),
#     Document(...company.pdf page 1...),
#     Document(...company.pdf page 2...),
#     ...
# ]
#
docs = text_docs + pdf_docs


# =========================================================
# 8. PRINT THE LOADED DOCUMENTS
# =========================================================

# This prints the complete list of Document objects.
#
# Each Document contains:
#
# Document(
#     page_content="actual document text",
#     metadata={
#         "source": "file path",
#         ...
#     }
# )
#
print(docs)