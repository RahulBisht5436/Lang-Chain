import os
import sys
from pathlib import Path
from typeDict_structuredOutout import ReviewAnalysis
import streamlit as st
from langchain_core.prompts import PromptTemplate
current_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(current_dir))

# Importing the base Model
from pydantic import BaseModel , Field , EmailStr
from typing import Optional
from llm.openAI_llm import llm


class Student(BaseModel):
    name:str 
    year:str = "first" #This is how we can set the default value 
    phone: str | None = Field(
        default="0000000000",
        min_length=10,
        max_length=10,
        pattern=r"^[6-9]\d{9}$"
    )
    
student = Student(name="Rahul Bisht" )
print(student)
