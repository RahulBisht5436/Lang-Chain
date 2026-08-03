import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from llm.groq_llm import generateResponse, llm
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug

load_dotenv()


set_debug(True)

genericPrompt = PromptTemplate(
    input_variables=['tech','no_of_paras'],
    template="""
    you are a CTO 
    you need to tell about the application of {tech} in breif
    in the no of paras {no_of_paras}
    
    Negative Cases: 
    
    Excluded Topics 
    
    Prompt Injection
    
    
    """
)


# ==============================>>>> UI layer
st.title("🤖 AI CTO ")

userEnteredTech = st.text_input("Mention the tech")
no_of_paras = st.number_input("Enter the number of Para Required", min_value=1,
    max_value=20,
    value=1,
    step=1)
if userEnteredTech :
    response = llm.invoke(genericPrompt.format(tech=userEnteredTech,no_of_paras=no_of_paras))
    st.write(response.content)