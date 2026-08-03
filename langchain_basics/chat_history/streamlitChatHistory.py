import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

# ChatPromptTemplate -> Used for chat models (ChatOllama, ChatOpenAI, etc.)
# MessagesPlaceholder -> Placeholder where previous conversation history will be injected
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm.ollama_llm import llm

# StreamlitChatMessageHistory
# Stores the chat history inside Streamlit's session state.
# Each browser session gets its own chat history.
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# RunnableWithMessageHistory
# Wrapper around an existing chain that automatically:
# 1. Reads previous messages
# 2. Injects them into the prompt
# 3. Saves the latest Human + AI messages after execution
from langchain_core.runnables.history import RunnableWithMessageHistory


# ------------------------------------------------------------------
# Prompt Template
# ------------------------------------------------------------------
# This prompt contains:
# 1. System message
# 2. Previous conversation (chat_history)
# 3. Current user question
#
# Final prompt sent to the LLM becomes something like:
#
# System : You are CTO...
#
# Human : Hi
# AI     : Hello
# Human : Explain Microservices
# AI     : ....
# Human : Design YouTube
#
prompt_template = ChatPromptTemplate(
    [
        ("system", "You are CTO of a organisation and need to provide HLD for a system"),

        # RunnableWithMessageHistory automatically injects previous
        # HumanMessage and AIMessage objects here.
        MessagesPlaceholder(variable_name="chat_history"),

        # Current user input
        ("human", "{input}")
    ]
)


# ------------------------------------------------------------------
# LCEL Chain
# ------------------------------------------------------------------
# Prompt ---> LLM
#
# ChatPromptTemplate
#          │
#          ▼
#      ChatOllama
#
chain = prompt_template | llm


# ------------------------------------------------------------------
# Visualize the chain structure
# ------------------------------------------------------------------
#
# Output:
#
# +--------------------+
# | ChatPromptTemplate |
# +--------------------+
#           │
#           ▼
# +--------------------+
# |     ChatOllama     |
# +--------------------+
#
chain.get_graph().print_ascii()


# ------------------------------------------------------------------
# Chat History Storage
# ------------------------------------------------------------------
#
# Stores messages inside Streamlit Session State.
#
# Session State:
#
# {
#     "chat_history": [
#          HumanMessage(...),
#          AIMessage(...),
#          HumanMessage(...)
#     ]
# }
#
history_of_chains = StreamlitChatMessageHistory(
    key="chat_history"
)


# ------------------------------------------------------------------
# RunnableWithMessageHistory
# ------------------------------------------------------------------
#
# This wraps the existing chain and automatically manages history.
#
# Every invocation performs:
#
# Step 1:
# Call lambda(session_id)
#
#            │
#            ▼
# Return history_of_chains
#
#
# Step 2:
# Read previous messages
#
#
# Step 3:
# Inject those messages into
# MessagesPlaceholder("chat_history")
#
#
# Step 4:
# Execute Prompt -> LLM
#
#
# Step 5:
# Automatically append
#
# HumanMessage(current input)
# AIMessage(current response)
#
# into chat history.
#
chain_with_history = RunnableWithMessageHistory(

    # Original chain
    chain,

    # Function that returns the history object.
    #
    # In production this usually returns different histories
    # based on session_id.
    #
    # Example:
    #
    # session_id = "abc123"
    # return history1
    #
    # session_id = "xyz789"
    # return history2
    #
    lambda session_id: history_of_chains,

    # Current user input variable
    input_messages_key="input",

    # Placeholder variable inside ChatPromptTemplate
    history_messages_key="chat_history"
)


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.title("HLD Generator")

# User enters system name
input = st.text_input("What System you need to have discussion on")


if input:

    # --------------------------------------------------------------
    # Invoke the chain
    # --------------------------------------------------------------
    #
    # Input:
    #
    # {
    #     "input": "Design Netflix"
    # }
    #
    #
    # Config:
    #
    # {
    #     "configurable": {
    #         "session_id": "abc123"
    #     }
    # }
    #
    #
    # Flow:
    #
    # User Input
    #      │
    #      ▼
    # RunnableWithMessageHistory
    #      │
    #      ├── Read previous history
    #      │
    #      ├── Inject into Prompt
    #      │
    #      ▼
    # ChatPromptTemplate
    #      │
    #      ▼
    # ChatOllama
    #      │
    #      ▼
    # AI Response
    #      │
    #      ▼
    # Save Human + AI messages back into history
    #
    result = chain_with_history.invoke(
        {"input": input},
        config={
            "configurable": {
                # Unique conversation identifier
                "session_id": "abc123"
            }
        }
    )

    # Display only the AI response
    st.write(result.content)