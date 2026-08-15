# Import Path for handling file and directory paths
# in a platform-independent way.
from pathlib import Path

# Import sys so that we can modify Python's module search path.
import sys

# Import the @tool decorator from LangChain.
#
# The @tool decorator converts our Python function into
# a LangChain Tool that can be called by an LLM/agent.
from langchain_community.tools import tool


# -------------------------------------------------------------------
# PROJECT PATH CONFIGURATION
# -------------------------------------------------------------------

# Get the project's root directory.
#
# __file__
#     -> Represents the current Python file.
#
# resolve()
#     -> Converts the path into an absolute path.
#
# parents[2]
#     -> Moves two directories up from the current file.
current_dir = Path(__file__).resolve().parents[2]


# Add the project root directory to Python's module search path.
#
# This allows us to import our own project modules, for example:
#
#     from llm.openAI_llm import llm
#     from postgresIntiation import db
#
# without getting a ModuleNotFoundError.
sys.path.insert(0, str(current_dir))


# -------------------------------------------------------------------
# LANGCHAIN IMPORTS
# -------------------------------------------------------------------

# PromptTemplate is used to create a reusable prompt
# containing dynamic variables such as:
#
#     {query}
#     {schema}
from langchain_core.prompts import PromptTemplate


# Import the LLM instance configured in our project.
#
# This LLM will be responsible for converting the user's
# natural-language request into a PostgreSQL SQL query.
from llm.openAI_llm import llm


# Import the PostgreSQL database connection.
#
# We use this connection to retrieve the schema of the
# students table.
from postgresIntiation import db


# StrOutputParser converts the LLM's response into
# a normal Python string.
#
# Without this parser, the LLM may return an AIMessage object.
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------------------------
# DATABASE SCHEMA
# -------------------------------------------------------------------

# Retrieve the schema information of the "students" table.
#
# The schema can contain information such as:
#
# - Column names
# - Data types
# - Primary keys
# - Constraints
# - Other table metadata
#
# This schema will be provided to the LLM so that it knows
# which columns are available when generating the SQL query.
schema = db.get_table_info(["students"])


# -------------------------------------------------------------------
# PROMPT TEMPLATE
# -------------------------------------------------------------------

# Create a prompt template that converts a user's
# natural-language request into a PostgreSQL SQL query.
#
# {query}
#     -> The request provided by the user/LLM.
#
# {schema}
#     -> The database schema that the LLM should use
#        while generating the SQL query.
promt = PromptTemplate(
    template="""
    Give Postgres SQL query for the implementation of
    {query}

    Only the read operation is allowed.

    Do NOT generate:
    - INSERT
    - UPDATE
    - DELETE
    - DROP
    - ALTER
    - TRUNCATE

    Only SELECT queries are allowed.

    The schema of the table is:
    {schema}

    Output Format:
    SELECT * FROM students;

    Nothing else.
    Return only the SQL query.
    """,

    # These variables will be replaced when
    # chain.invoke() is called.
    input_variables=["query", "schema"]
)


# -------------------------------------------------------------------
# OUTPUT PARSER
# -------------------------------------------------------------------

# Create an output parser.
#
# The LLM generally returns an AIMessage.
# StrOutputParser extracts only the text content
# from the LLM response.
parser = StrOutputParser()


# -------------------------------------------------------------------
# LANGCHAIN CHAIN
# -------------------------------------------------------------------

# Create the LangChain processing chain.
#
# The pipe operator "|" connects each component.
#
# Flow:
#
# User Query
#      ↓
# PromptTemplate
#      ↓
# LLM
#      ↓
# StrOutputParser
#      ↓
# SQL Query String
#
chain = promt | llm | parser


# -------------------------------------------------------------------
# SQL GENERATION TOOL
# -------------------------------------------------------------------

# Convert the Python function into a LangChain Tool.
#
# Once decorated with @tool, this function can be provided
# to an LLM/agent as a callable tool.
@tool
def getSQLQuery(query: str) -> str:
    """
    Generate a read-only PostgreSQL SQL query from a
    natural-language request.

    The tool uses the students table schema to generate
    a valid SQL query.

    Only SELECT/read operations are allowed.
    INSERT, UPDATE, DELETE, DROP, ALTER, and TRUNCATE
    operations must not be generated.

    Args:
        query: Natural-language request describing the
               data the user wants to retrieve.

    Returns:
        A PostgreSQL SELECT query as a string.
    """

    # Send the user's request and database schema
    # to the LangChain chain.
    #
    # "query"  -> User's natural-language request.
    # "schema" -> PostgreSQL students table schema.
    queryString = chain.invoke({
        "query": query,
        "schema": schema
    })

    # Return the generated SQL query.
    return queryString


@tool
def runQuery(query):
    """ This Tool is used to return the Database after executing the {query} operation """
    results = db.run(query)
    return results