import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from llm.ollama_llm import llm

prompt_template = ChatPromptTemplate(
    [
        ("system" , "You are CTO of a organisation and need to provide HLD for a system" ),
        ("human", "{input}")
    ]
)


st.title("HLD Generator")
input = st.text_input("What System you need to have discussion on ")

if input : 
    result = llm.invoke(prompt_template.format(input=input))
    st.write(result.content)