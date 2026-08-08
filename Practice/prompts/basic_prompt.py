import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------
# IMPORT LLM
# ---------------------------------------------------------

from llm.openAI_llm import llm

# ---------------------------------------------------------
# PROMPT FILE PATH
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_PATH = os.path.join(
    CURRENT_DIR,
    "template.json"
)

print("Prompt path:", PROMPT_PATH)
print("Prompt exists:", os.path.exists(PROMPT_PATH))

# ---------------------------------------------------------
# LOAD PROMPT
# ---------------------------------------------------------

template = load_prompt(PROMPT_PATH)

# ---------------------------------------------------------
# CREATE CHAIN
# ---------------------------------------------------------

chain = template | llm

# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="LangChain Chat",
    page_icon="🤖"
)

st.title("🤖 LangChain Chatbot")

# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# DISPLAY OLD MESSAGES
# ---------------------------------------------------------

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

question = st.chat_input("Ask something...")

if question:

    with st.chat_message("user"):
        st.write(question)

    # -----------------------------------------------------
    # CONVERT HISTORY TO TEXT
    # -----------------------------------------------------

    history_text = "\n".join(
        [
            f"User: {chat['question']}\n"
            f"Assistant: {chat['answer']}"
            for chat in st.session_state.chat_history
        ]
    )

    # -----------------------------------------------------
    # CALL LLM
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = chain.invoke(
                {
                    "chatHistory": history_text,
                    "question": question
                }
            )

            response = answer.content

        st.write(response)

    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": response
        }
    )