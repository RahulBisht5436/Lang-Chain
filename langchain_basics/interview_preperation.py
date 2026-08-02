import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from ollama_llm import llm

# =========================================================
# WHAT THIS APP DOES
# =========================================================
# User enters a topic (example: "Docker").
# Then 3 LLM steps run one after another (a chain):
#   1) Create a simple definition
#   2) Expand that definition into a detailed explanation
#   3) Use the detailed explanation to make interview Q&A
#
# Pattern used:
#   PromptTemplate | llm
# This means: fill the prompt with variables, then send it to the model.


# -----------------------------
# Prompt 1 : Definition
# -----------------------------
# input_variables=["topic"] means {topic} in the template must be provided.
definition_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are an expert teacher.

Give a simple and easy-to-understand definition of the following topic.

Topic:
{topic}
"""
)

# -----------------------------
# Prompt 2 : Detailed Explanation
# -----------------------------
# This prompt does NOT take the original topic.
# It takes the DEFINITION produced by Prompt 1.
detail_prompt = PromptTemplate(
    input_variables=["definition"],
    template="""
You are an expert instructor.

The following is the definition of a topic:

{definition}

Using this definition, explain the topic in great detail.

Include:
1. What it is
2. Why it is used
3. How it works
4. Advantages
5. Disadvantages
6. Real-world examples
7. Best practices
8. Interview tips
"""
)

# -----------------------------
# Prompt 3 : Interview Questions
# -----------------------------
# This prompt takes the DETAILED EXPLANATION from Prompt 2.
interview_prompt = PromptTemplate(
    input_variables=["details"],
    template="""
You are a senior technical interviewer.

The candidate has studied the following topic:

{details}

Generate the Top 10 interview questions.

For each question also provide a detailed answer.
"""
)

# -----------------------------
# Chain 1: Topic -> Definition
# -----------------------------
# "|" connects steps into a pipeline.
# So: fill definition_prompt, then call llm.
definition_chain = definition_prompt | llm

# -----------------------------
# Chain 2: Definition -> Detailed Explanation
# -----------------------------
# After Chain 1, llm returns an AIMessage object (not plain text).
# detail_prompt expects a dict like: {"definition": "..."}.
# RunnableLambda converts AIMessage -> that dict.
detail_chain = (
    RunnableLambda(
        # definition is the AIMessage from the previous llm call
        lambda definition: {
            "definition": definition.content  # extract only the text
        }
    )
    | detail_prompt  # put that text into {definition}
    | llm            # ask the model for a detailed explanation
)

detail_chain.get_graph().print_ascii()
# -----------------------------
# Chain 3: Details -> Interview Q&A
# -----------------------------
# Same idea as Chain 2:
# convert AIMessage -> {"details": "..."} -> prompt -> llm
interview_chain = (
    RunnableLambda(
        lambda details: {
            "details": details.content
        }
    )
    | interview_prompt
    | llm
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 Topic Analyzer")

topic = st.text_input("Enter a topic")

if st.button("Analyze"):

    if topic:

        with st.spinner("Generating..."):

            # Step 1: pass {"topic": "..."} into definition_chain
            # Output is an AIMessage with .content = short definition
            definition = definition_chain.invoke(
                {"topic": topic}
            )
            
            

            st.header("Definition")
            st.write(definition.content)

            # Step 2: pass the AIMessage from Step 1 into detail_chain
            # RunnableLambda extracts .content and builds {"definition": ...}
            details = detail_chain.invoke(
                definition
            )

            st.header("Detailed Explanation")
            st.write(details.content)

            # Step 3: pass the AIMessage from Step 2 into interview_chain
            # RunnableLambda extracts .content and builds {"details": ...}
            interview = interview_chain.invoke(
                details
            )

            st.header("Interview Questions")
            st.write(interview.content)

    else:
        st.warning("Please enter a topic.")
