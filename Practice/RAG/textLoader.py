from langchain_community.document_loaders import TextLoader
from pathlib import Path


# ---------------------------------------------------------
# 1. Get the project directory
# ---------------------------------------------------------

current_dir = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# 2. Define the file path
# ---------------------------------------------------------

file_path = current_dir / "RAG" / "Data" / "company.txt"

print("=" * 80)
print("FILE PATH")
print("=" * 80)
print(file_path)


# ---------------------------------------------------------
# 3. Create the TextLoader
# ---------------------------------------------------------

loader = TextLoader(str(file_path))


# ---------------------------------------------------------
# 4. Load the document
# ---------------------------------------------------------

docs = loader.load()


# ---------------------------------------------------------
# 5. Display document information
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DOCUMENT INFORMATION")
print("=" * 80)

print(f"Number of documents loaded: {len(docs)}")


# ---------------------------------------------------------
# 6. Display each Document
# ---------------------------------------------------------

for index, doc in enumerate(docs, start=1):

    print("\n" + "=" * 80)
    print(f"DOCUMENT {index}")
    print("=" * 80)

    # Metadata
    print("\n--- METADATA ---")
    print(doc.metadata)

    # Individual metadata values
    print("\n--- SOURCE ---")
    print(doc.metadata.get("source"))

    # Page content
    print("\n--- PAGE CONTENT ---")
    print(doc.page_content)

    # Content length
    print("\n--- CONTENT LENGTH ---")
    print(len(doc.page_content))