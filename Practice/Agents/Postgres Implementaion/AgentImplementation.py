import streamlit as st
from pathlib import Path
import sys


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

# Get the project root directory.
#
# __file__
#     -> Current Python file
#
# resolve()
#     -> Converts the path to an absolute path
#
# parents[2]
#     -> Moves two directories up
current_dir = Path(__file__).resolve().parents[2]


# Add the project root to Python's module search path.
#
# This allows us to import our project modules such as:
#
#     from llm.openAI_llm import llm
#     from postgresImplementation import getSQLQuery, runQuery
sys.path.insert(0, str(current_dir))


# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

# AgentExecutor and create_react_agent are legacy/classic
# LangChain agent APIs.
#
# We import them from langchain_classic instead of
# langchain.agents because newer LangChain versions
# moved the classic APIs into langchain-classic.
from langchain_classic.agents import (
    AgentExecutor,
    create_react_agent
)

# PromptTemplate for the standard ReAct prompt.
# (hub.pull was removed from langchain 1.x; define locally instead.)
from langchain_core.prompts import PromptTemplate


# ============================================================
# LLM
# ============================================================

# Import your configured LLM.
from llm.ollama_llm import llm


# ============================================================
# TOOLS
# ============================================================

# getSQLQuery:
#     Converts the user's natural-language question
#     into a PostgreSQL SQL query.
#
# runQuery:
#     Executes the generated SQL query against PostgreSQL.
from postgresImplementation import (
    getSQLQuery,
    runQuery
)


# ============================================================
# REACT PROMPT
# ============================================================

# Standard ReAct prompt (same as hwchase17/react on LangChain Hub).
#
# ReAct means:
#
#     Reason
#       ↓
#     Action
#       ↓
#     Observation
#       ↓
#     Reason
#       ↓
#     Action
#       ↓
#     Observation
#       ↓
#     Final Answer
#
react_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


# ============================================================
# TOOLS AVAILABLE TO THE AGENT
# ============================================================

# Give the ReAct agent access to both tools.
#
# The agent can decide when to call each tool.
tools = [
    getSQLQuery,
    runQuery
]


# ============================================================
# CREATE REACT AGENT
# ============================================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)


# ============================================================
# CREATE AGENT EXECUTOR
# ============================================================

agent_executor = AgentExecutor(
    agent=agent,

    # Tools that the executor is allowed to execute.
    tools=tools,

    # --------------------------------------------------------
    # VERBOSE
    # --------------------------------------------------------
    #
    # True means LangChain will print the agent's
    # intermediate execution steps in the terminal.
    #
    # Example:
    #
    # > Entering new AgentExecutor chain...
    #
    # Thought: I need to generate SQL.
    #
    # Action: getSQLQuery
    #
    # Action Input: Give me the highest CGPA scorer
    #
    # Observation:
    # SELECT * FROM students ...
    #
    # Thought: Now I need to execute the query.
    #
    # Action: runQuery
    #
    # Action Input: SELECT * FROM students ...
    #
    # Observation:
    # [(46, 'Student30', ...)]
    #
    # Final Answer:
    # Student30 has the highest CGPA.
    #
    # > Finished chain
    verbose=True,

    # If the LLM generates something that the ReAct
    # parser cannot understand, allow the agent to
    # try again instead of immediately failing.
    # handle_parsing_errors=True,

    # Maximum number of agent reasoning/action iterations.
    # This prevents the agent from running indefinitely.
    max_iterations=50,

    # Return tool calls so we can display the generated SQL in the UI.
    return_intermediate_steps=True
)


# ============================================================
# HELPERS
# ============================================================

def extract_sql_query(intermediate_steps) -> str | None:
    """Pull the generated SQL from the agent's tool-call history."""
    for action, observation in intermediate_steps:
        if action.tool == "getSQLQuery":
            return str(observation).strip()
        if action.tool == "runQuery":
            return str(action.tool_input).strip()
    return None


def extract_query_results(intermediate_steps) -> str | None:
    """Pull the database rows returned by runQuery."""
    for action, observation in intermediate_steps:
        if action.tool == "runQuery":
            return str(observation).strip()
    return None


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PostgreSQL AI Assistantass",
    page_icon="🗄️",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🗄️ PostgreSQL AI Assistant")

st.write(
    "Ask a question about the students database in natural language. "
    "The ReAct Agent will generate SQL, execute it, and return the result."
)


# ============================================================
# USER INPUT
# ============================================================

query = st.text_input(
    "What is your query?",
    placeholder="Example: Give me the highest CGPA scorer"
)


# ============================================================
# EXECUTE AGENT
# ============================================================

if st.button("Run Query"):

    # Check whether the user entered a query.
    if not query.strip():

        st.warning("Please enter a query.")

    else:

        try:

            # ------------------------------------------------
            # RUN REACT AGENT
            # ------------------------------------------------

            with st.spinner(
                "Agent is generating SQL and querying PostgreSQL..."
            ):

                response = agent_executor.invoke(
                    {
                        "input": query
                    }
                )


            # ------------------------------------------------
            # DISPLAY SQL QUERY
            # ------------------------------------------------

            sql_query = extract_sql_query(
                response.get("intermediate_steps", [])
            )
            query_results = extract_query_results(
                response.get("intermediate_steps", [])
            )

            if sql_query:
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")

            if query_results:
                st.subheader("Query Result")
                st.code(query_results, language="text")

            # ------------------------------------------------
            # DISPLAY FINAL RESULT
            # ------------------------------------------------

            st.subheader("Agent Result")

            st.code(
                str(response["output"]),
                language="text"
            )


        except Exception as e:

            # ------------------------------------------------
            # DISPLAY ERROR
            # ------------------------------------------------

            st.error(
                f"Agent Error: {e}"
            )
            
AgentFinish(
    return_values={
        "output": "The average CGPA is 7.8."
    },
    log="The required information has been obtained."
)
