from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)