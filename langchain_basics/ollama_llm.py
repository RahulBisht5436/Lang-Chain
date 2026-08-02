from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def chatWithOllama(ques):
    return   llm.invoke(ques) 
