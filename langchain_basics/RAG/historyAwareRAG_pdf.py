# ==========================================================
# Import Required Libraries
# ==========================================================

# Embedding model that converts text into vectors for similarity search.
from llm.ollamaEmbedding import embeddings

# Chat LLM (Ollama) used to rewrite questions and generate answers.
from llm.ollama_llm import llm

# Loads a plain .txt file into LangChain Document objects.
from langchain_community.document_loaders import PyPDFLoader

# Splits long documents into smaller overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma vector store: indexes chunks and retrieves similar ones.
from langchain_chroma import Chroma

# ChatPromptTemplate builds prompts; MessagesPlaceholder injects chat history.
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

# RAG chain helpers:
# - create_history_aware_retriever: rewrites follow-ups using chat history,
#   then retrieves relevant docs
# - create_retrieval_chain: wires retriever -> QA chain end-to-end
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

# Stuff-documents chain: inserts retrieved docs into {context}, then calls the LLM.
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# In-memory store for Human / AI messages per session.
from langchain_community.chat_message_histories.in_memory import (
    ChatMessageHistory,
)

# Wraps a chain so it automatically loads/saves chat history each turn.
from langchain_core.runnables.history import RunnableWithMessageHistory

from pathlib import Path

# ==========================================================
# Prompt 1 — Contextualize / rewrite the question
# ==========================================================
# Goal: turn a vague follow-up like "Which one is cheapest?"
# into a standalone question using chat_history, e.g.
# "Which product from the catalog has the lowest price?"
#
# This rewritten question is used ONLY for retrieval (search),
# not as the final answer.
# ==========================================================

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are given a chat history and the latest user question.

Rewrite the latest question so that it can be understood
without the previous conversation.

Do NOT answer the question.

If the question is already standalone,
return it unchanged.
""",
        ),
        # Previous Human/AI messages for this session.
        MessagesPlaceholder("chat_history"),
        # Latest user question.
        ("human", "{input}"),
    ]
)

# ==========================================================
# Prompt 2 — Answer using retrieved documents
# ==========================================================
# {context} is filled with retrieved document chunks.
# {input} is the user's question (original, not necessarily rewritten).
# chat_history helps the model stay consistent across turns.
# ==========================================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an assistant for answering questions.

Use ONLY the retrieved context below to answer.

If the answer is not present in the context,
say:

"I don't know based on the provided context."

Keep your answer concise.

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ==========================================================
# 1. Load Documents
# ==========================================================

current_dir = Path(__file__).parent
file_path = current_dir / "resume.pdf"

# Load product.txt as Document object(s).
documents = PyPDFLoader(str(file_path)).load()

# ==========================================================
# 2. Split Documents
# ==========================================================
# Smaller chunks improve retrieval for specific facts (prices, specs).
# Overlap keeps related sentences from being cut mid-thought.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

# ==========================================================
# 3. Create Vector Store + Base Retriever
# ==========================================================
# Embed each chunk and store vectors in Chroma (in-memory here).

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# Base retriever: finds top similar chunks for a query string.
retriever = db.as_retriever()

# ==========================================================
# 4. History-Aware Retriever
# ==========================================================
# Flow:
#   chat_history + input
#     -> LLM rewrites question (using contextualize_q_prompt)
#     -> base retriever searches with the rewritten question
#     -> returns relevant Document chunks

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# ==========================================================
# 5. Question-Answer Chain
# ==========================================================
# Takes retrieved docs + question (+ history) and produces an answer.

qa_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

# ==========================================================
# 6. Complete RAG Chain
# ==========================================================
# End-to-end:
#   input (+ chat_history)
#     -> history_aware_retriever (get docs)
#     -> qa_chain (generate answer)
# Returns a dict with keys: "input", "context", "answer"

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)

# ==========================================================
# 7. Session Chat History Store
# ==========================================================
# Maps session_id -> ChatMessageHistory.
# Each session keeps its own conversation separately.

store = {}


def get_session_history(session_id: str):
    """Return (or create) the message history for a given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ==========================================================
# 8. Wrap RAG Chain with Message History
# ==========================================================
# RunnableWithMessageHistory automatically:
#   - loads past messages into chat_history before invoke
#   - appends the new Human + AI messages after invoke
#
# Keys:
#   input_messages_key   -> where the user text lives in the input dict
#   history_messages_key -> prompt variable that receives past messages
#   output_messages_key  -> which result field to save as the AI reply
#                          (retrieval chains use "answer", not "output")

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ==========================================================
# 9. Interactive Chat Loop
# ==========================================================

print("=" * 60)
print("History Aware RAG Chat")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    # session_id groups messages into one conversation.
    # Change it (or make it per-user) to start a fresh history.
    result = chain_with_history.invoke(
        {
            "input": question,
        },
        config={
            "configurable": {
                "session_id": "abc123",
            }
        },
    )

    # result is a dict; print only the generated answer text.
    print("\nAI :", result["answer"])
