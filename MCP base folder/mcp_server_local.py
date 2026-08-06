# =============================================================================
# Imports
# =============================================================================

# Library to fetch summaries from Wikipedia
import wikipedia

# DuckDuckGo Search API
from duckduckgo_search import DDGS

# FastMCP framework for creating an MCP Server
from mcp.server.fastmcp import FastMCP


# =============================================================================
# Create MCP Server
# =============================================================================

# Create a FastMCP server instance.
# The name is displayed to MCP clients during tool discovery.
mcp = FastMCP(name="Tool Server")


# =============================================================================
# Wikipedia Search Tool
# =============================================================================

# Register this function as an MCP Tool.
# Any MCP Client connected to this server can invoke this tool.
@mcp.tool()
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and return a short summary.

    Args:
        query: Topic to search.

    Returns:
        A short summary from Wikipedia.
    """

    try:
        # Retrieve a concise summary (first 2 sentences)
        summary = wikipedia.summary(query, sentences=2)

        return summary

    # Raised when multiple Wikipedia pages match the query
    except wikipedia.exceptions.DisambiguationError as e:

        # Show only the first few matching options
        options = ", ".join(e.options[:5])

        return (
            f"Your query is ambiguous.\n"
            f"Possible matches: {options}"
        )

    # Raised when no matching page exists
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this query."

    # Catch any unexpected errors
    except Exception as e:
        return f"Wikipedia Error: {e}"


# =============================================================================
# DuckDuckGo Search Tool
# =============================================================================

# Register this function as another MCP Tool.
@mcp.tool()
def ddg_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query.

    Returns:
        Top search results.
    """

    try:

        # Create a DuckDuckGo search session
        with DDGS() as ddgs:

            # Fetch the top 3 search results
            results = list(ddgs.text(query, max_results=3))

            # Handle empty results
            if not results:
                return "No search results found."

            output = []

            # Format each result into a readable string
            for i, result in enumerate(results, start=1):

                # Extract available fields safely
                title = result.get("title", "No Title")
                body = result.get("body", "")
                href = result.get("href", "")

                output.append(
                    f"{i}. {title}\n"
                    f"{body}\n"
                    f"{href}"
                )

            # Return all formatted results
            return "\n\n".join(output)

    # Catch any unexpected errors
    except Exception as e:
        return f"DuckDuckGo Error: {e}"


# =============================================================================
# Start the MCP Server
# =============================================================================

# This block executes only when this file is run directly.
# It does NOT execute when this module is imported elsewhere.
if __name__ == "__main__":

    # Start the MCP Server using Streamable HTTP transport.
    #
    # Default endpoint:
    #     http://localhost:8000/mcp
    #
    # The server listens for incoming MCP Client connections.
    # Clients can:
    #   1. Discover available tools
    #   2. Read tool schemas
    #   3. Invoke tools remotely
    #
    # Other available transports:
    #   transport="stdio"   -> Local subprocess communication
    #   transport="sse"     -> Server-Sent Events (legacy)
    #   transport="streamable-http" -> Modern HTTP-based MCP transport
    mcp.run(transport="stdio")