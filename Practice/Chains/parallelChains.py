import os
import sys
from pathlib import Path


# =========================================================
# Import LangChain Components
# =========================================================

# StrOutputParser converts the LLM's AIMessage output
# into a simple Python string.
from langchain_core.output_parsers import StrOutputParser


# RunnableParallel allows multiple chains to execute
# independently using the same input.
#
# Example:
#
# Input
#   ↓
# ┌───────────────┬───────────────┐
# ↓                               ↓
# General Chain                 HLD Chain
# ↓                               ↓
# Output                        Output
#
from langchain_core.runnables import RunnableParallel


# PromptTemplate is used to create reusable prompts
# containing variables such as {topic}.
from langchain_core.prompts import PromptTemplate


# =========================================================
# Add Project Root Directory to Python Path
# =========================================================

# __file__ = current Python file
#
# parents[1] moves two levels up in the directory structure.
#
# This allows us to import project modules such as:
#
# from llm.openAI_llm import llm

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


# =========================================================
# Import LLM
# =========================================================

# Import the LLM configured in our project.
from llm.openAI_llm import llm


# =========================================================
# 1. General Information Prompt
# =========================================================

# This prompt generates general information about the topic.
#
# Input:
#
# {
#     "topic": "LangChain"
# }
#
# The {topic} variable will be replaced with the
# value provided at runtime.

template_general = PromptTemplate(
    template="""
Provide the general information about the topic.

The topic is {topic}
""",
    input_variables=["topic"]
)


# =========================================================
# 2. HLD (High-Level Design) Prompt
# =========================================================

# This prompt generates the High-Level Design (HLD)
# for the given topic.
#
# It also expects the same {topic} input.

template_HLD = PromptTemplate(
    template="""
Provide the High-Level Design (HLD) for the topic.

The topic is {topic}
""",
    input_variables=["topic"]
)


# =========================================================
# 3. Output Parser
# =========================================================

# StrOutputParser converts the LLM response into
# a plain Python string.
#
# Without StrOutputParser:
#
#     LLM → AIMessage
#
# With StrOutputParser:
#
#     LLM → String

parser = StrOutputParser()


# =========================================================
# 4. Create Parallel Chains
# =========================================================

# RunnableParallel executes both chains independently
# using the SAME input.
#
# Chain 1:
#
# template_general → LLM → parser
#
# Chain 2:
#
# template_HLD → LLM → parser
#
# Both chains receive:
#
# {
#     "topic": "LangChain"
# }
#
# The results are stored using these keys:
#
#     generalInformation
#     HLDInformation
#
# The output of parallelChain will look like:
#
# {
#     "generalInformation": "...",
#     "HLDInformation": "..."
# }

parallelChain = RunnableParallel(
    {
        "generalInformation":
            template_general | llm | parser,

        "HLDInformation":
            template_HLD | llm | parser
    }
)


# =========================================================
# 5. Merge Prompt
# =========================================================

# After the parallel chains finish, we have two outputs:
#
#     generalInformation
#     HLDInformation
#
# This prompt receives both outputs and asks the LLM
# to combine them into one comprehensive answer.

merge_prompt = PromptTemplate(
    template="""
You are given two pieces of information about a topic.

General Information:
{generalInformation}

HLD Information:
{HLDInformation}

Combine these two pieces of information into one
comprehensive and well-structured answer.
""",
    input_variables=[
        "generalInformation",
        "HLDInformation"
    ]
)


# =========================================================
# 6. Create Final Merged Chain
# =========================================================

# Complete flow:
#
# Input
#   ↓
# RunnableParallel
#   ├── General Information → LLM → Parser
#   └── HLD Information → LLM → Parser
#              ↓
#        Dictionary
#              ↓
#        Merge Prompt
#              ↓
#             LLM
#              ↓
#           Parser
#              ↓
#        Final String

mergedChain = (
    parallelChain
    | merge_prompt
    | llm
    | parser
)


# =========================================================
# 7. Display Chain Graph
# =========================================================

# get_graph() returns the runnable graph of the chain.
#
# print_ascii() displays the chain structure
# in the terminal.

mergedChain.get_graph().print_ascii()


# =========================================================
# 8. Get Topic From User
# =========================================================

topic = input("What is the topic? ")


# =========================================================
# 9. Invoke the Chain
# =========================================================

if topic:

    # The input must be a dictionary because
    # both PromptTemplates expect the variable {topic}.
    #
    # The same input is automatically passed to
    # both branches of RunnableParallel.

    results = mergedChain.invoke(
        {
            "topic": topic
        }
    )


    # =====================================================
    # 10. Print Final Result
    # =====================================================

    # results contains the final output after:
    #
    # Parallel execution
    #       ↓
    # Merge prompt
    #       ↓
    # LLM
    #       ↓
    # StrOutputParser

    print("\n================ FINAL RESULT ================\n")

    print(results)