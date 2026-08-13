# =========================================================
# Python Standard Library Imports
# =========================================================

import sys
from pathlib import Path


# =========================================================
# LangChain Imports
# =========================================================

# PromptTemplate is used to create a reusable prompt
# containing variables such as {question}.
from langchain_core.prompts import PromptTemplate


# =========================================================
# Add Parent Directory to Python Module Path
# =========================================================

# __file__
#   → Path of the current Python file.
#
# Path(__file__).resolve()
#   → Converts the file path into an absolute path.
#
# .parents[1]
#   → Goes two levels up from the current file.
#
# str(...)
#   → Converts the Path object into a normal string.
#
# sys.path.insert(0, ...)
#   → Adds that directory to Python's module search path.
#
# This allows us to import modules from our project's
# parent directories.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


# =========================================================
# Import LLM
# =========================================================

# Import the LLM object from our custom LLM configuration.
#
# The llm object is expected to be an instance of a
# LangChain-compatible chat model.
from llm.openAI_llm import llm


# =========================================================
# 1. CREATE PROMPT TEMPLATE
# =========================================================

# PromptTemplate allows us to create a reusable prompt
# with dynamic input variables.
#
# {question} is a placeholder.
# Its actual value will be provided when we call:
#
# chain.stream({
#     "question": question
# })
#
prompt = PromptTemplate(
    template="""
    Answer the following question clearly and concisely.

    Question:
    {question}
    """,

    # Tell LangChain which variables exist inside
    # the prompt template.
    input_variables=["question"],
)


# =========================================================
# 2. CREATE CHAIN
# =========================================================

# The "|" operator is LangChain Expression Language (LCEL).
#
# It connects multiple Runnable components together.
#
# Flow:
#
# User Question
#       ↓
# PromptTemplate
#       ↓
# LLM
#       ↓
# AI Response
#
# In other words:
#
# prompt | llm
#
# means:
#
# First execute the prompt,
# then pass its output to the LLM.
chain = prompt | llm


# =========================================================
# 3. DEFINE TAGS
# =========================================================

# Tags are useful for tracing and observability.
#
# For example, when using LangSmith or another
# tracing/monitoring system, these tags can help
# identify what type of chain execution occurred.
#
# Here we are saying:
#
# production
#   → This execution belongs to production-like code.
#
# question_answer
#   → This chain is being used for question answering.
tagList = [
    "production",
    "question_answer"
]


# =========================================================
# 4. GET USER INPUT
# =========================================================

# input() pauses the program and waits for the user
# to enter a question in the terminal.
#
# Example:
#
# Enter your question: What is Python?
#
# The entered text is stored inside the "question"
# variable.
question = input(
    "Enter your question: "
)


# =========================================================
# 5. CHECK FOR EMPTY INPUT
# =========================================================

# .strip() removes spaces from the beginning and end.
#
# Example:
#
# "   ".strip()
# → ""
#
# If the user enters only spaces or nothing meaningful,
# we don't execute the LLM chain.
if not question.strip():

    print("Please enter a question.")

else:

    # =====================================================
    # 6. STREAM THE CHAIN RESPONSE
    # =====================================================

    # stream() executes the chain and returns the response
    # incrementally as chunks.
    #
    # Unlike:
    #
    # chain.invoke(...)
    #
    # which waits for the complete response,
    #
    # stream()
    # starts giving us pieces of the response as they
    # become available.
    #
    # Example:
    #
    # LLM response:
    #
    # "Python is a programming language..."
    #
    # stream() may produce chunks such as:
    #
    # "Python"
    # " is"
    # " a"
    # " programming"
    # " language"
    #
    # Exact chunk behavior depends on the LLM.
    for chunk in chain.stream(

        # This dictionary provides the value for
        # {question} in PromptTemplate.
        {
            "question": question
        },

        # RunnableConfig contains additional configuration
        # for this particular chain execution.
        config={

            # Tags help identify and categorize this
            # execution in tracing/observability systems.
            "tags": tagList,

            # Metadata is additional information attached
            # to this execution.
            #
            # This can be useful for:
            #
            # - Logging
            # - Tracing
            # - Debugging
            # - Monitoring
            # - Analytics
            "metadata": {
                "type": "question_answer"
            }
        }
    ):

        # chunk contains one piece of the streamed response.
        #
        # For a typical ChatModel response:
        #
        # chunk.content
        #
        # contains the actual text generated by the model.
        #
        # end=""
        #   → Don't add a new line after every chunk.
        #
        # flush=True
        #   → Immediately display the chunk in the terminal
        #      instead of waiting for the output buffer.
        #
        # Together, these make the response appear like
        # ChatGPT is typing the answer in real time.
        print(
            chunk.content,
            end="",
            flush=True
        )


    # =====================================================
    # 7. PRINT ANSWER
    # =====================================================

    # We don't need to print the answer here because
    # every chunk has already been printed above.
    #
    # If we were using invoke(), we could do:
    #
    # result = chain.invoke(...)
    #
    # print(result.content)
    #
    # But with stream(), the response is printed
    # incrementally inside the for-loop.
