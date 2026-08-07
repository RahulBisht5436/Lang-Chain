from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=2
)
question = input("What is your question : ")
if(question):
    answer = llm.invoke(
        question
    )
    print(answer.content)