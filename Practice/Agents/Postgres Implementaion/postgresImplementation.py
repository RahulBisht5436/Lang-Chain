# Import Path for handling file/directory paths in a platform-independent way
from pathlib import Path

# Import sys so that we can modify Python's module search path
import sys


# Get the project's root directory.
#
# __file__ -> current Python file
# resolve() -> converts it to an absolute path
# parents[2] -> moves two directories up from the current file
current_dir = Path(__file__).resolve().parents[2]

# Add the project root directory to Python's module search path.
#
# This allows us to import modules from our project, such as:
#     from llm.openAI_llm import llm
#     from postgresIntiation import db
sys.path.insert(0, str(current_dir))


# PromptTemplate is used to create a reusable prompt
# with placeholders such as {query} and {schema}
from langchain_core.prompts import PromptTemplate

# Import the LLM instance that we configured in our project
from llm.openAI_llm import llm

# Import the PostgreSQL database connection
from postgresIntiation import db

# StrOutputParser converts the LLM's output into a plain Python string
from langchain_core.output_parsers import StrOutputParser


# Get information about the "students" table from PostgreSQL.
#
# This generally contains information such as:
# - Column names
# - Data types
# - Constraints
# - Other table metadata
#
# We will provide this schema information to the LLM so that
# it can generate SQL queries based on the actual database structure.
schema = db.get_table_info(["students"])


# Create a prompt template for converting natural-language questions
# into PostgreSQL SQL queries.
#
# {query}  -> User's question
# {schema} -> Database schema
promt = PromptTemplate(
    template="""
    Give Postgres SQL query for the implementation of
    {query}

    Only the read operation is allowed.
    Block any UPDATE, DELETE, or INSERT query execution.

    The schema of the table is:
    {schema}
    """,

    # These are the variables that will be replaced when
    # chain.invoke() is called.
    input_variables=["query", "schema"]
)


# Create an output parser.
#
# The LLM normally returns an AIMessage object.
# StrOutputParser extracts the actual text content from it.
parser = StrOutputParser()


# Create a LangChain chain using the pipe operator.
#
# Flow:
#
# User Query
#     ↓
# PromptTemplate
#     ↓
# LLM
#     ↓
# StrOutputParser
#     ↓
# SQL query as a string
#
chain = promt | llm | parser


# Take a question from the user through the terminal.
#
# Example:
# "Show me all students whose CGPA is greater than 8"
query = input("How may I help you? ")


# Check whether the user entered something.
#
# An empty string evaluates to False.
if query:

    # Send the user's question and database schema to the chain.
    #
    # The values replace:
    #     {query}  -> user's question
    #     {schema} -> PostgreSQL table schema
    #
    # The chain then:
    # 1. Builds the prompt
    # 2. Sends it to the LLM
    # 3. Parses the LLM response into a string
    queryString = chain.invoke({
        "query": query,
        "schema": schema
    })

    # Print the generated SQL query.
    print(queryString)