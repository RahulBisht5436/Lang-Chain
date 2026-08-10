import os
import sys
from pathlib import Path

# StrOutputParser converts the LLM's response into a simple Python string.
from langchain_core.output_parsers import StrOutputParser

# PromptTemplate is used to create a reusable prompt with variables.
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------
# STEP 1: Find the project's root/current directory
# ---------------------------------------------------------

# __file__ represents the path of the current Python file.
#
# Path(__file__) converts that path into a Path object.
#
# .resolve() converts it into an absolute path.
#
# .parents[1] goes two levels upward in the directory structure.
#
# Example:
#
# Project/
# ├── Agents/
# │   └── weatherAgent.py   <-- current file
# └── llm/
#     └── openAI_llm.py
#
# parents[1] can be used to reach the Project directory.
current_dir = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# STEP 2: Add project directory to Python's import path
# ---------------------------------------------------------

# Python normally searches for modules in certain predefined
# directories.
#
# By adding current_dir to sys.path, Python will also search
# inside our project directory when importing modules.
sys.path.insert(0, str(current_dir))


# ---------------------------------------------------------
# STEP 3: Import our configured LLM
# ---------------------------------------------------------

# This "llm" object was configured separately in our project.
#
# For example, openAI_llm.py might contain:
#
# llm = ChatOpenAI(...)
#
# We import that already-configured LLM here so that we can
# use it inside our LangChain chain.
from llm.openAI_llm import llm


# ---------------------------------------------------------
# STEP 4: Create an output parser
# ---------------------------------------------------------

# StrOutputParser converts the LLM's response into a plain
# Python string.
#
# Without the parser, depending on the LLM being used,
# the result may be an AIMessage object.
parser = StrOutputParser()


# ---------------------------------------------------------
# STEP 5: Create the prompt template
# ---------------------------------------------------------

promptWeather = PromptTemplate(
    template="""
You need to provide activities that can be performed in {city}
given the current weather is {weather}.
""",

    # These are the variables that must be provided
    # when invoking the prompt.
    input_variables=["city", "weather"]
)


# ---------------------------------------------------------
# STEP 6: Create the LangChain chain
# ---------------------------------------------------------

# The | operator connects multiple LangChain components.
#
# Data flows from left to right:
#
# PromptTemplate
#       ↓
#      LLM
#       ↓
# StrOutputParser
#
# So:
#
# User input
#     ↓
# PromptTemplate creates the final prompt
#     ↓
# LLM generates an answer
#     ↓
# StrOutputParser converts the answer to a string
chainWeatherActivity = promptWeather | llm | parser


# ---------------------------------------------------------
# STEP 7: Create a reusable function
# ---------------------------------------------------------

def get_weather_activity(city: str, weather: str) -> str:
    """
    Suggest activities to do in a city based on its
    current weather.

    Args:
        city: The name of the city.
        weather: A description of the current weather
                 in that city.

    Returns:
        A string containing activities suggested by the LLM.
    """

    # Invoke the complete LangChain chain.
    #
    # The values here replace the variables in the prompt:
    #
    # {city}    → city
    # {weather} → weather
    answer = chainWeatherActivity.invoke({
        "city": city,
        "weather": weather
    })

    # Return the final LLM response as a normal Python string.
    return answer