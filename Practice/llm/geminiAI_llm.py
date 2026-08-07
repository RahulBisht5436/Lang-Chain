from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # Fast and inexpensive
    temperature=0.5,
)

def generate_response(question: str):
    response = llm.invoke(question)
    return response.content

