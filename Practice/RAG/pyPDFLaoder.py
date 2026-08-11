# Import PyPDFLoader to load and extract text from PDF files
from langchain_community.document_loaders import PyPDFLoader

# Import Path for handling file and directory paths
from pathlib import Path


# Get the project's root/current directory.
# __file__ represents the current Python file.
# parents[1] moves two levels up from the current file's directory.
current_dir = Path(__file__).resolve().parents[1]


def PDFLoader(fileName: str):
    """
    Load a PDF file from the RAG/Data directory.

    Args:
        fileName (str): Name of the PDF file, for example:
                        "company.pdf"

    Returns:
        list: A list of LangChain Document objects.
              Each document generally represents one page of the PDF.
    """

    # Build the complete path to the PDF file:
    # <project_root>/RAG/Data/<fileName>
    file_path = current_dir / "RAG" / "Data" / fileName

    # Create a PDF loader for the specified file
    loader = PyPDFLoader(str(file_path))
    print(file_path,"File Path from the PyPDF Laoder===========================>>>>")
    # Load the PDF and extract its content.
    # PyPDFLoader returns a list of Document objects,
    # usually one Document object per PDF page.
    docs = loader.load()

    # Return the extracted documents so they can be
    # processed further, for example for text splitting or RAG.
    return docs