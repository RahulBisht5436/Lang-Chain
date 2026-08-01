from calendar import month
from tkinter import Variable
import dotenv
import os
import streamlit as st
from groq_llm import llm
from langchain_core.prompts import PromptTemplate


travelPrompt = PromptTemplate(
    input_variables=["city","months","language","budget"],
    template="""
    greet with the message 
    Welcome to the {city} Travel Guide 
    if you are visiting in {months} here is what you can do 
    
    then provide the 
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    """
)


st.title("Hello , this is your travel guide")
city = st.text_input("In which City")
month = st.number_input("For How many months",min_value=0,step=1)
language = st.text_input("What is you language")
budget = st.number_input("What is you Budget",min_value=0)


if city and budget and language and month : 
    response= llm.invoke(travelPrompt.format(city=city,budget=budget, language=language,months=month))
    st.write(response.content)