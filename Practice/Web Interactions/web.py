import os
import sys
from pathlib import Path
from tavily import TavilyClient
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



# Practice/ project root must be on sys.path before importing local packages
current_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(current_dir))


from llm.openAI_llm import llm


tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

prompt = PromptTemplate(
    template = """
                    You are a helpful AI assistant.

                    Answer the user's question using ONLY the web
                    search results provided below.

                    If the search results do not contain enough
                    information, say that you don't have enough
                    information.

                    User Question:
                    {Topic}

                    Web Search Results:
                    {web_results}
               """,
    input_variables=["Topic", "web_results"]
)

parser = StrOutputParser()

topic = input("what is the topic \n")
search_results = tavily.search(
    query=topic,
    search_depth="advanced",
    max_results=3
)

web_results = "\n\n".join(
    [
        f"Title: {result['title']}\n"
        f"URL: {result['url']}\n"
        f"Content: {result['content']}"
        for result in search_results["results"]
    ]
)
chain = prompt | llm | parser

if topic : 
    result = chain.invoke({"Topic":topic,"web_results": web_results})
    print("\nANSWER:\n")
    print(result)

