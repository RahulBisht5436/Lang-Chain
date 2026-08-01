from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="gemma4:12b",
    temperature=0
)

def chatWithOllama(ques):
    return   llm.invoke(ques) 
