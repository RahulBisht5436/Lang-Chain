# ============================================================
# WEB SEARCH IMPLEMENTATION USING LANGCHAIN TOOLS
# ============================================================

# `tool` is a decorator used to convert a normal Python
# function into a LangChain Tool.
#
# `DuckDuckGoSearchRun` is a built-in LangChain tool that
# allows us to perform web searches using DuckDuckGo.

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# CREATE THE BUILT-IN DUCKDUCKGO SEARCH TOOL
# ============================================================

# Create an instance of the built-in DuckDuckGo search tool.
#
# This object already knows how to communicate with
# DuckDuckGo and perform a search.

searchEngine = DuckDuckGoSearchRun()


# ============================================================
# IMPORTANT RECOMMENDATIONS FOR CUSTOM TOOLS
# ============================================================

# When creating a custom tool, follow these recommendations:
#
# 1. Add a DOCSTRING
#    -> Tell the LLM what the tool does.
#
# 2. Use TYPE HINTING
#    -> Tell LangChain what type of input the tool expects
#       and what type of output it returns.
#
# 3. Use the @tool DECORATOR
#    -> Converts a normal Python function into a LangChain Tool.
#
# Example:
#
# @tool
# def addition(a: int, b: int) -> int:
#     """Add two numbers and return the result."""
#     return a + b


# ============================================================
# 1. BUILT-IN TOOL
# ============================================================

def searchExternal(query: str) -> str:
    """
    Search the internet using DuckDuckGo
    and return the search results.

    Parameters:
        query (str):
            The search question/query provided by the user.

    Returns:
        str:
            Search results returned by DuckDuckGo.
    """

    # `.invoke()` executes the LangChain tool.
    #
    # We pass the user's search query to DuckDuckGo.
    result = searchEngine.invoke(query)

    # Return the result instead of only printing it.
    # Returning is important because another component,
    # such as an Agent or LLM, may need to consume this result.
    return result


# ============================================================
# TEST THE BUILT-IN SEARCH TOOL
# ============================================================

print("Result from the built-in tool:")

result = searchExternal("What date is today?")

print(result)


# ============================================================
# 2. CUSTOM TOOL
# ============================================================

# `@tool` converts this normal Python function into
# a LangChain Tool.
#
# Once converted, an Agent/LLM can be given this tool
# and can decide when to call it.

@tool
def addition(a: int, b: int) -> int:
    """
    Add two numbers and return their sum.

    Parameters:
        a (int):
            First number.

        b (int):
            Second number.

    Returns:
        int:
            Sum of a and b.
    """

    # Perform the actual action of the tool.
    return a + b


# ============================================================
# TEST THE CUSTOM TOOL
# ============================================================

# Because `addition` is now a LangChain Tool,
# we can execute it using `.invoke()`.

result = addition.invoke({
    "a": 10,
    "b": 20
})

print("Addition Result:", result)