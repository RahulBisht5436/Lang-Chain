from dotenv import load_dotenv
from groq_llm import generateResponse, llm
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug

load_dotenv()


set_debug(True)

genericPrompt = PromptTemplate(
    input_variables=['tech'],
    template="""
    you are a CTO 
    you need to tell about the application of {tech} in breif
    """
)


# ==============================>>>> UI layer
st.title("🤖 AI CTO ")

userEnteredTech = st.text_input("Mention the tech")
if userEnteredTech :
    response = llm.invoke(genericPrompt.format(tech=userEnteredTech))
    st.write(response.content)