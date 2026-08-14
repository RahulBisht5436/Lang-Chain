import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from pinecone import Pinecone


# =========================================================
# Configuration
# =========================================================

# Load environment variables from the .env file.
#
# Your .env file should contain:
#
# PINECONE_API_KEY=your-api-key
#
load_dotenv()


# Read the Pinecone API key from environment variables.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Stop the program if the API key is not available.
if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set. "
        "Add it to your .env file."
    )


# ---------------------------------------------------------
# Pinecone configuration
# ---------------------------------------------------------

# Name of the Pinecone index where our vectors/records
# will be stored.
INDEX_NAME = "company-knowledge"


# Namespace is used to logically separate records
# inside the same Pinecone index.
#
# Think of it like a folder/category inside the index.
NAMESPACE = "company-documents"


# ---------------------------------------------------------
# PDF configuration
# ---------------------------------------------------------

# Path to the PDF file.
#
# __file__ represents the current Python file.
#
# Path(__file__).parent
#       ↓
# directory containing this Python file
#
# / "Data" / "company_data.pdf"
#       ↓
# Data/company_data.pdf
#
PDF_PATH = Path(__file__).parent / "Data" / "company_data.pdf"


# ---------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------

# Maximum number of characters in one chunk.
CHUNK_SIZE = 1000


# Number of characters that should overlap between
# consecutive chunks.
#
# Example:
#
# Chunk 1 -> characters 0 - 999
# Chunk 2 -> characters 800 - 1799
#
# Therefore, 200 characters are shared between chunks.
#
# Overlap helps prevent important information from being
# lost at chunk boundaries.
CHUNK_OVERLAP = 200


# =========================================================
# Initialize Pinecone
# =========================================================

# Create a Pinecone client using the API key.
#
# The client is responsible for communicating with
# Pinecone's cloud service.
pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =========================================================
# Create Index
# =========================================================

# Check whether the index already exists.
#
# If it does not exist, create it.
if not pc.has_index(INDEX_NAME):

    print(f"Creating Pinecone index: {INDEX_NAME}")


    # -----------------------------------------------------
    # Create an index with integrated embedding
    # -----------------------------------------------------
    #
    # We are using Pinecone's integrated embedding model:
    #
    # llama-text-embed-v2
    #
    # This means we DON'T have to manually generate
    # embeddings using a separate embedding model.
    #
    # Pinecone receives our text and generates the
    # vector representation internally.
    #
    pc.create_index_for_model(
        name=INDEX_NAME,

        # Cloud provider where Pinecone will host
        # the index.
        cloud="aws",

        # AWS region where the index will be created.
        region="us-east-1",

        # Configuration for integrated embedding.
        embed={

            # Embedding model that Pinecone will use
            # to convert text into vectors.
            "model": "llama-text-embed-v2",

            # field_map tells Pinecone which field in our
            # records contains the text that needs to be
            # converted into an embedding.
            #
            # Our records will contain:
            #
            # {
            #     "_id": "...",
            #     "chunk_text": "some text"
            # }
            #
            # Therefore:
            #
            # "text" -> "chunk_text"
            #
            # means Pinecone should generate embeddings
            # from the "chunk_text" field.
            "field_map": {
                "text": "chunk_text"
            }
        }
    )


    print("Index creation started...")


else:

    # The index already exists, so we don't create it again.
    print(f"Index already exists: {INDEX_NAME}")


# =========================================================
# Wait for Index
# =========================================================

print("Waiting for Pinecone index...")


# Index creation may take some time.
#
# We repeatedly check whether the index is ready.
while True:

    # Get information about the index.
    description = pc.describe_index(INDEX_NAME)


    # Check the ready status of the index.
    #
    # When True, we can start uploading records.
    if description.status.ready:
        break


    # Index is still being created.
    print("Index is not ready yet...")


    # Wait for 2 seconds before checking again.
    time.sleep(2)


print("Pinecone index is ready.")


# =========================================================
# Read PDF
# =========================================================

def extract_pdf_text(pdf_path):

    """
    Extract text from every page of the PDF.

    Returns a list containing:
    
    [
        {
            "page": 1,
            "text": "..."
        },
        {
            "page": 2,
            "text": "..."
        }
    ]
    """

    print(f"Reading PDF: {pdf_path}")


    # Open/read the PDF.
    reader = PdfReader(pdf_path)


    # This list will contain the extracted text
    # from each page.
    pages = []


    # Iterate through every page in the PDF.
    #
    # enumerate() gives us:
    #
    # page_number -> 0, 1, 2, ...
    # page        -> actual PDF page object
    #
    for page_number, page in enumerate(reader.pages):


        # Extract text from the current page.
        text = page.extract_text()


        # Some PDF pages may contain no extractable text.
        #
        # Only add the page if text was successfully extracted.
        if text:

            pages.append(
                {
                    # PDF page numbers should start from 1
                    # instead of Python's 0-based indexing.
                    "page": page_number + 1,

                    # Actual extracted text.
                    "text": text
                }
            )


    # Return all extracted pages.
    return pages


# =========================================================
# Chunk Text
# =========================================================

def create_chunks(pages):

    """
    Split the extracted PDF text into smaller chunks.

    Each chunk contains:
    
    - unique ID
    - text
    - page number
    """

    # List that will contain all chunks.
    chunks = []


    # Counter used to generate unique chunk IDs.
    chunk_id = 0


    # Process every page individually.
    for page in pages:

        # Get text from the current page.
        text = page["text"]


        # Starting position for the first chunk.
        start = 0


        # Continue creating chunks until we reach
        # the end of the page.
        while start < len(text):


            # Calculate the end position of the chunk.
            #
            # Example:
            #
            # start = 0
            # CHUNK_SIZE = 1000
            #
            # end = 1000
            end = start + CHUNK_SIZE


            # Extract the chunk.
            chunk_text = text[start:end]


            # Remove unnecessary whitespace.
            chunk_text = chunk_text.strip()


            # Only create a record if the chunk
            # actually contains text.
            if chunk_text:

                chunks.append(
                    {
                        # Create a unique ID for the chunk.
                        #
                        # Example:
                        #
                        # company-page-1-chunk-0
                        # company-page-1-chunk-1
                        #
                        "id": (
                            f"company-page-{page['page']}"
                            f"-chunk-{chunk_id}"
                        ),

                        # Actual text of the chunk.
                        "text": chunk_text,

                        # Store the original PDF page number.
                        #
                        # This becomes metadata in Pinecone
                        # and can later help us identify
                        # where the information came from.
                        "page": page["page"]
                    }
                )


                # Increment the global chunk ID.
                chunk_id += 1


            # Move to the next chunk.
            #
            # We subtract CHUNK_OVERLAP so that part of
            # the previous chunk is repeated.
            #
            # Example:
            #
            # CHUNK_SIZE = 1000
            # CHUNK_OVERLAP = 200
            #
            # First chunk:
            # 0 -> 999
            #
            # Next chunk:
            # 800 -> 1799
            #
            # Therefore, characters 800-999 overlap.
            start += CHUNK_SIZE - CHUNK_OVERLAP


    # Return all generated chunks.
    return chunks


# =========================================================
# Load PDF
# =========================================================

# Before reading the PDF, make sure the file actually exists.
if not PDF_PATH.exists():

    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )


# Extract text from the PDF.
pages = extract_pdf_text(PDF_PATH)


# Display the number of pages successfully processed.
print(f"Pages extracted: {len(pages)}")


# =========================================================
# Create Chunks
# =========================================================

# Convert the extracted pages into smaller chunks.
chunks = create_chunks(pages)


# Display the total number of chunks created.
print(f"Total chunks created: {len(chunks)}")


# =========================================================
# Prepare Pinecone Records
# =========================================================

# Pinecone will receive records instead of raw PDF pages.
records = []


# Convert every chunk into the format expected by
# Pinecone's upsert_records() method.
for chunk in chunks:

    records.append(
        {
            # Unique ID for this record.
            #
            # Pinecone uses this to identify the record.
            "_id": chunk["id"],


            # IMPORTANT:
            #
            # This is the field Pinecone will use to generate
            # the embedding because of:
            #
            # field_map = {
            #     "text": "chunk_text"
            # }
            #
            "chunk_text": chunk["text"],


            # Metadata: original PDF page number.
            "page": chunk["page"],


            # Metadata: original source file.
            "source": PDF_PATH.name
        }
    )


# =========================================================
# Upsert into Pinecone
# =========================================================

# Create a handle/reference to our Pinecone index.
index = pc.Index(INDEX_NAME)


print("Uploading chunks to Pinecone...")


# Number of records uploaded in one request.
#
# Instead of sending hundreds/thousands of records
# in a single request, we send them in smaller batches.
BATCH_SIZE = 50


# Iterate through the records in batches.
#
# Example:
#
# 0 - 49
# 50 - 99
# 100 - 149
# ...
for i in range(0, len(records), BATCH_SIZE):


    # Select the current batch.
    batch = records[i:i + BATCH_SIZE]


    # Upload the batch to Pinecone.
    #
    # namespace:
    #     Logical grouping of our records.
    #
    # records:
    #     The actual text + metadata.
    #
    # Because this index uses integrated embedding,
    # Pinecone will generate embeddings from the
    # "chunk_text" field automatically.
    index.upsert_records(
        namespace=NAMESPACE,
        records=batch
    )


    # Display upload progress.
    #
    # min() prevents the displayed number from going
    # beyond the actual number of records.
    print(
        f"Uploaded {min(i + BATCH_SIZE, len(records))}"
        f"/{len(records)} chunks"
    )


# =========================================================
# Completion Message
# =========================================================

print()
print("======================================")
print("PDF ingestion completed successfully!")
print("======================================")


# Display important information about the ingestion.
print(f"Index     : {INDEX_NAME}")
print(f"Namespace : {NAMESPACE}")
print(f"Chunks    : {len(records)}")