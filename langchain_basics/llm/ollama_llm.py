from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load .env from langchain_basics/ regardless of where the app is run from
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def chatWithOllama(ques):
    return llm.invoke(ques)
