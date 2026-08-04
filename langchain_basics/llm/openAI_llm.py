from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = ChatOpenAI(
    model="gpt-5-nano",      # Cheapest model
    temperature=0.5,
)

def generateResponse(question: str, max_completion_tokens: int = 4096):
    response = llm.invoke(
        question,
        max_completion_tokens=max_completion_tokens
    )
    return response.content

if __name__ == "__main__":
    response = generateResponse(
        "I am in love, but the funny thing is she is not in love with me."
    )
    print(response)