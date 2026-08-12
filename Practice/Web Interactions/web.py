import os
import sys
from pathlib import Path

from tavily import TavilyClient

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. ADD PROJECT ROOT TO PYTHON PATH
# ============================================================
#
# __file__ represents the path of the current Python file.
#
# Path(__file__).resolve()
#     -> Gives the absolute path of this Python file.
#
# .parents[1]
#     -> Moves two levels upward from the current file.
#
# This is useful when your project has a structure like:
#
# Practice/
# ├── llm/
# │   └── openAI_llm.py
# │
# └── RAG/
#     └── Web/
#         └── web_search.py
#
# We want Python to be able to find:
#
#     from llm.openAI_llm import llm
#
# Therefore, we add the Practice/ directory to sys.path.
# ============================================================

current_dir = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(current_dir))


# ============================================================
# 2. IMPORT YOUR LLM
# ============================================================
#
# This is your existing LLM configuration.
#
# For example, openAI_llm.py may contain:
#
#     llm = ChatOpenAI(...)
#
# The LLM itself DOES NOT search the Internet.
#
# Tavily will perform the web search and we will provide
# the search results to this LLM.
# ============================================================

from llm.openAI_llm import llm


# ============================================================
# 3. INITIALIZE TAVILY CLIENT
# ============================================================
#
# Tavily is responsible for searching the Internet.
#
# The API key should be stored as an environment variable:
#
#     TAVILY_API_KEY
#
# Example:
#
#     TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxx
#
# Do NOT hard-code the API key directly into your Python file.
# ============================================================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ============================================================
# 4. CREATE THE PROMPT
# ============================================================
#
# We are telling the LLM:
#
# "Do not answer from your own knowledge.
#  Use the information retrieved from the web."
#
# {Topic}
#     -> User's question
#
# {web_results}
#     -> Information retrieved from Tavily
# ============================================================

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


# ============================================================
# 5. OUTPUT PARSER
# ============================================================
#
# The LLM may return an AIMessage object depending on
# the LLM implementation.
#
# StrOutputParser converts the LLM output into a simple
# Python string.
#
# Example:
#
# AIMessage(content="OpenAI announced...")
#
# becomes:
#
# "OpenAI announced..."
# ============================================================

parser = StrOutputParser()


# ============================================================
# 6. GET USER QUESTION
# ============================================================
#
# input() allows the user to enter a question from
# the terminal.
#
# Example:
#
#     What is the latest OpenAI news today?
# ============================================================

topic = input("What is the topic?\n")


# ============================================================
# 7. SEARCH THE WEB USING TAVILY
# ============================================================
#
# This is the important part.
#
# Tavily sends the user's question to its web-search
# service and retrieves current information from the web.
#
# search_depth="advanced"
#     -> Performs a deeper search.
#
# max_results=3
#     -> We are requesting a maximum of 3 search results.
#
# The returned object will contain information such as:
#
#     title
#     url
#     content
#
# Example:
#
# {
#     "results": [
#         {
#             "title": "...",
#             "url": "...",
#             "content": "..."
#         }
#     ]
# }
# ============================================================

if topic:

    search_results = tavily.search(
        query=topic,
        search_depth="advanced",
        max_results=3
    )


    # ========================================================
    # 8. EXTRACT WEB RESULTS
    # ========================================================
    #
    # Tavily returns multiple search results.
    #
    # We convert those results into one large string so that
    # we can pass them to the LLM as context.
    #
    # For every result we include:
    #
    #     Title
    #     URL
    #     Content
    #
    # Example:
    #
    # Title: OpenAI announces...
    # URL: https://example.com/...
    # Content: OpenAI announced...
    #
    # --------------------------------------------------------
    #
    # "\n\n".join(...)
    #
    # combines all search results and separates them with
    # two newline characters.
    # ========================================================

    web_results = "\n\n".join(
        [
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content']}"
            for result in search_results["results"]
        ]
    )


    # ========================================================
    # 9. CREATE LANGCHAIN CHAIN
    # ========================================================
    #
    # The pipe operator "|" connects the components:
    #
    #     prompt
    #       ↓
    #     llm
    #       ↓
    #     parser
    #
    # So the complete flow is:
    #
    #     PromptTemplate
    #          ↓
    #         LLM
    #          ↓
    #     StrOutputParser
    #
    # ========================================================

    chain = prompt | llm | parser


    # ========================================================
    # 10. SEND WEB RESULTS + USER QUESTION TO LLM
    # ========================================================
    #
    # We now invoke the chain.
    #
    # Topic:
    #     The question asked by the user.
    #
    # web_results:
    #     Fresh information retrieved from Tavily.
    #
    # The prompt receives:
    #
    #     {Topic}        -> topic
    #     {web_results}  -> web_results
    #
    # ========================================================

    result = chain.invoke({
        "Topic": topic,
        "web_results": web_results
    })


    # ========================================================
    # 11. DISPLAY FINAL ANSWER
    # ========================================================
    #
    # The LLM has now generated an answer using the
    # information retrieved from the Internet.
    # ========================================================

    print("\nANSWER:\n")
    print(result)