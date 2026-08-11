import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Add the project root directory to Python's module search path.
#
# parents[2] points to the project root directory.
# This allows Python to find and import our custom RAG package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Import our custom PDFLoader function.
#
# PDFLoader:
# 1. Loads the PDF file.
# 2. Extracts the text from each page.
# 3. Returns the extracted content as a list of
#    LangChain Document objects.
from RAG.pyPDFLaoder import PDFLoader


# Create a RecursiveCharacterTextSplitter.
#
# RecursiveCharacterTextSplitter divides large documents
# into smaller chunks while trying to preserve meaningful
# text boundaries.
#
# chunk_size:
# Maximum number of characters allowed in a chunk.
#
# chunk_overlap:
# Number of characters that are repeated between
# consecutive chunks.
#
# Example:
#
# Chunk 1:
# "Python is a programming language used for..."
#
# Chunk 2:
# "...language used for building applications..."
#
# The overlapping text helps preserve context between
# consecutive chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)


def textSpillterCustom(file_path: str):
    """
    Load a PDF and split its content into smaller chunks.

    Args:
        file_path (str):
            Name or path of the PDF file to be loaded.

    Returns:
        List[Document]:
            A list of LangChain Document objects,
            where each Document represents one text chunk.
    """

    # Load the PDF using our custom PDFLoader.
    #
    # PDFLoader returns:
    # List[Document]
    #
    # Each Document contains:
    # - page_content -> extracted text
    # - metadata     -> information such as source and page number
    docs = PDFLoader(file_path)


    # Split the loaded Document objects into smaller chunks.
    #
    # split_documents() takes:
    #     List[Document]
    #
    # and returns:
    #     List[Document]
    #
    # The metadata of the original documents is also
    # preserved in the generated chunks.
    splitted_docs = splitter.split_documents(docs)


    # Return the list of generated document chunks.
    return splitted_docs


# Execute this block only when this Python file
# is run directly.
#
# It will NOT execute when this file is imported
# as a module from another Python file.
if __name__ == "__main__":

    # Load the PDF and split it into smaller chunks.
    splitted_docs = textSpillterCustom("company.pdf")


    # Iterate through every generated document chunk.
    for doc in splitted_docs:

        # Print the actual text contained inside the chunk.
        print(doc.page_content)

        # Print metadata associated with the chunk.
        #
        # Example:
        # {
        #     'source': 'company.pdf',
        #     'page': 0
        # }
        print(doc.metadata)

        # Print a separator between chunks
        # to make the terminal output easier to read.
        print("-" * 50)
