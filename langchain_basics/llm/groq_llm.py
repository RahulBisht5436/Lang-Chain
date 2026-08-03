from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env from langchain_basics/ regardless of where the app is run from
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=.5
)

def generateResponse(question):
    pass
    response = llm.invoke(question)
    return response


