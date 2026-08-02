from math import factorial
from ollama_llm import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.runnables import RunnableLambda , RunnablePassthrough

seachPrompt = PromptTemplate(
    template="""
    You need to write a powerful speech of 350 words
    on the {topics}
    """,
    input_variables=["topics"] 
)

factsPrompt = PromptTemplate(
    template="""
    provide 5 Facts from about thr
    on the {text}
    """,
    input_variables=["text"] 
)


speech_chain = seachPrompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(
        speech=lambda x: speech_chain.invoke(x)
    )
    | RunnablePassthrough.assign(
        facts=lambda x: (
            factsPrompt
            | llm
            | StrOutputParser()
        ).invoke({"text": x["speech"]})
    )
)

st.title("Welcome to Speach Generator")
ques= st.text_input("Which Topic due you want to talk about ")
if ques: 
    result = chain.invoke({
        "topics": ques
    })

    st.subheader("Speech")
    st.write(result["speech"])

    st.subheader("Facts")
    st.write(result["facts"])

