# Import CSVLoader from LangChain.
# CSVLoader is used to read data from a CSV file
# and convert each row into a LangChain Document object.
from langchain_community.document_loaders import CSVLoader

# Import Path to work with file and directory paths
# in a platform-independent way.
from pathlib import Path


# Get the directory where the current Python file (CSVLoader.py) is located.
#
# __file__ → path of the current Python file
# resolve() → converts it into an absolute path
# parent → gets the directory containing the current file
#
# Example:
# Practice/RAG/CSVLoader.py
#              ↓ parent
# Practice/RAG/
current_dir = Path(__file__).resolve().parent


# Create the complete path to the CSV file.
#
# current_dir → Practice/RAG/
# "Data"      → Data folder
# "employees.csv" → CSV file
#
# Final path:
# Practice/RAG/Data/employees.csv
file_path = current_dir / "Data" / "employees.csv"


# Print the complete file path.
# This is useful for debugging and checking whether
# Python is looking at the correct location.
print(str(file_path), "=============================>>>>>>>")


# Create a CSVLoader object and provide the CSV file path.
#
# str(file_path) converts the Path object into a string,
# which is accepted by CSVLoader.
loader = CSVLoader(str(file_path))


# Load the CSV file.
#
# load() reads the CSV file and converts each row
# into a LangChain Document object.
#
# The result is stored inside csv_docs as a list of Documents.
csv_docs = loader.load()


# Print all the loaded Document objects.
print(csv_docs)