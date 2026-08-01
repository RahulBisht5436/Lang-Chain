from dotenv import load_dotenv
from groq_llm import generateResponse
import streamlit as st
from langchain_core.globals import set_debug
load_dotenv()


set_debug(True)

st.title("🤖 AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box at the bottom
question = st.chat_input("Ask me anything...")

if question:
    # Show user's message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Get AI response
    response = generateResponse(question)

    answer = response.content

    # Show assistant's response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })