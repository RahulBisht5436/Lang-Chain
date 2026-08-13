import os
import sys
from pathlib import Path
import requests
import dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

current_dir = Path(__file__).resolve().parents[1]

dotenv.load_dotenv(current_dir / ".env")

sys.path.insert(0, str(current_dir))

from llm.openAI_llm import llm

SEARXNG_URL = os.getenv("SEARXNG_URL")

if not SEARXNG_URL:
    raise ValueError("SEARXNG_URL is not configured in .env")

prompt = PromptTemplate(
    template="""
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

topic = input("What is the topic?\n")


if topic:

    # search_results = tavily.search(
    #     query=topic,
    #     search_depth="advanced",
    #     max_results=3
    # )
    
    params = {
        "q": topic,
        "format": "json",
    }

    response = requests.get(
        f"{SEARXNG_URL}/search",
        params=params,
        timeout=10
    )

    response.raise_for_status()

    search_results = response.json()


    # ========================================================
    # 7. EXTRACT SEARCH RESULTS
    # ========================================================

    results = search_results.get("results", [])

    if not results:
        print("No web search results found.")
        sys.exit(0)


    web_results = "\n\n".join(
        [
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {result.get('content', 'N/A')}"
            for result in results[:5]
        ])
    # web_results = "\n\n".join(
    #     [
    #         f"Title: {result['title']}\n"
    #         f"URL: {result['url']}\n"
    #         f"Content: {result['content']}"
    #         for result in search_results["results"]
    #     ]
    # )


    chain = prompt | llm | parser


    result = chain.invoke({
        "Topic": topic,
        "web_results": web_results
    })


    print("\nANSWER:\n")
    print(result)