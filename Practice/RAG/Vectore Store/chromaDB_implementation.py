import sys
from pathlib import Path

from langchain_chroma import Chroma

# Practice/ project root must be on sys.path before importing local packages
current_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(current_dir))

from RAG.text_splitter.textBasedSplitting import textSpillterCustom
from llm.openaiEmbedding import embeddings

result = textSpillterCustom("company.pdf")


vectorstore = Chroma.from_documents(
    documents=result,
    embedding=embeddings,
    persist_directory="chroma_db_persist"
)

print("Chroma vector store created successfully!")
