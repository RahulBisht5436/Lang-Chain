import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Add the project root directory to Python's module search path.
# This allows Python to import custom modules/packages from your project.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2])
)

# Import Wikipedia API library.
# This library allows us to search and retrieve content from Wikipedia.
import wikipediaapi

# LangChain's Document class is used to represent retrieved information.
# A Document contains:
#   1. page_content -> actual text/content
#   2. metadata     -> additional information about the content
from langchain_core.documents import Document


# Create a Wikipedia API client.
#
# user_agent:
#   Identifies your application when making requests to Wikipedia.
#   It is recommended to provide your application name and a contact email.
#
# language="en":
#   Tells Wikipedia API that we want to work with English Wikipedia.
wiki = wikipediaapi.Wikipedia(
    user_agent="LangChainPractice/1.0 (your-email@example.com)",
    language="en"
)


# Take the topic from the user at runtime.
#
# Example:
#   What is the Topic?
#   Artificial Intelligence
#
# The value entered by the user will be used to search Wikipedia.
topic = input("What is the Topic?\n")


# Retrieve the Wikipedia page for the requested topic.
#
# For example:
#   topic = "Artificial Intelligence"
#
# page will contain information about the Wikipedia page,
# including its title, text, URL, etc.
page = wiki.page(topic)


# Convert the Wikipedia page into a LangChain Document.
#
# page_content:
#   Stores the actual Wikipedia article text.
#
# metadata:
#   Stores additional information about the document.
#   Here we are storing:
#       title  -> Wikipedia page title
#       source -> URL of the Wikipedia page
doc = Document(
    page_content=page.text,
    metadata={
        "title": page.title,
        "source": page.fullurl
    }
)


# Print the first 5000 characters of the retrieved Wikipedia content.
#
# [:5000] means:
#   Start from character 0
#   Stop at character 5000
#
# This prevents the entire Wikipedia article from being printed
# if the article is very large.
print(doc.page_content[:5000])