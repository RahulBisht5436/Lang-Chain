from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
