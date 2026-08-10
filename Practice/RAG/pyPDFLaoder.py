from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
current_dir = Path(__file__).resolve().parents[1]
file_path = current_dir / "RAG" / "Data" / "company.pdf"

loader = PyPDFLoader(str(file_path))
docs = loader.load()
print(docs)



