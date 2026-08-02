import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from ollama_llm import llm

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