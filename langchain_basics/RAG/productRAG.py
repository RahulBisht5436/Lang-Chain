# Embeddings convert text into vectors for similarity search in the vector DB.
from llm.ollamaEmbedding import embeddings

# Chat LLM (Ollama) that generates the final answer from retrieved context.
from llm.ollama_llm import llm

# TextLoader reads a plain .txt file into LangChain Document objects.
from langchain_community.document_loaders import TextLoader

# Splits long documents into smaller overlapping chunks for better retrieval.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma is the vector store that indexes and retrieves relevant chunks.
from langchain_chroma import Chroma

# ChatPromptTemplate builds the system + human messages sent to the LLM.
from langchain_core.prompts import ChatPromptTemplate

# RAG chain helpers:
# - create_stuff_documents_chain: stuffs retrieved docs into the prompt
# - create_retrieval_chain: wires retriever -> document chain end-to-end
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from pathlib import Path

# -----------------------------------------------------------------------------
# Prompt Template
# -----------------------------------------------------------------------------
# {context} is filled with retrieved document chunks.
# {input} is filled with the user's question.
# -----------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an assistant for answering questions.
                Use the provided context to respond.
                If the answer isn't clear, acknowledge that you don't know.
                Limit your response to three concise sentences.

                Context:
                {context}
            """
        ),
        ("human", "{input}")
    ]
)

# -----------------------------------------------------------------------------
# 1. Load the product catalog
# -----------------------------------------------------------------------------
current_dir = Path(__file__).parent
file_path = current_dir / "product.txt"

# Load product.txt as a list of Document objects (usually one big document).
documents = TextLoader(str(file_path)).load()

# -----------------------------------------------------------------------------
# 2. Split into chunks
# -----------------------------------------------------------------------------
# chunk_size: max characters per chunk
# chunk_overlap: shared characters between neighboring chunks (keeps context)
textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200
)

chunks = textSplitter.split_documents(documents)

# -----------------------------------------------------------------------------
# 3. Build the vector store
# -----------------------------------------------------------------------------
# Embed each chunk and store vectors in an in-memory Chroma collection.
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Retriever finds the most similar chunks for a given question.
retriver = db.as_retriever()

# -----------------------------------------------------------------------------
# 4. Build the RAG chain
# -----------------------------------------------------------------------------
# Stuff documents chain: inserts retrieved docs into {context}, then calls llm.
qa_chain = create_stuff_documents_chain(llm, prompt)

# Retrieval chain: question -> retrieve docs -> answer with qa_chain.
rag_chain = create_retrieval_chain(retriver, qa_chain)

# -----------------------------------------------------------------------------
# 5. Ask a question and print the answer
# -----------------------------------------------------------------------------
question = input("What do you want to ask ?? ")

# Returns a dict with keys: "input", "context", "answer"
result = rag_chain.invoke({"input": question})

print("\nAnswer:")
print(result["answer"])
