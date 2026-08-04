# ==========================================================
# Import Required Libraries
# ==========================================================

# Embedding model that converts text into vectors.
from llm.ollamaEmbedding import embeddings

# Ollama LLM
from llm.ollama_llm import llm

# Loads text documents
from langchain_community.document_loaders import TextLoader

# Splits documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store
from langchain_chroma import Chroma

# Prompt Templates
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

# Chains
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# Chat History
from langchain_community.chat_message_histories.in_memory import (
    ChatMessageHistory,
)

# Runnable History
from langchain_core.runnables.history import RunnableWithMessageHistory

from pathlib import Path

# ==========================================================
# Prompt 1
# Rewrites follow-up questions into standalone questions
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
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ==========================================================
# Prompt 2
# Answers using retrieved documents
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
# Load Documents
# ==========================================================

current_dir = Path(__file__).parent
file_path = current_dir / "product.txt"

documents = TextLoader(str(file_path)).load()

# ==========================================================
# Split Documents
# ==========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

# ==========================================================
# Create Vector Store
# ==========================================================

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

retriever = db.as_retriever()

# ==========================================================
# History Aware Retriever
# ==========================================================

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# ==========================================================
# Question Answer Chain
# ==========================================================

qa_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

# ==========================================================
# Complete RAG Chain
# ==========================================================

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)

# ==========================================================
# Store Chat Histories
# ==========================================================

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ==========================================================
# Chain with Message History
# ==========================================================

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
     output_messages_key="answer"
)

# ==========================================================
# Chat Loop
# ==========================================================

print("=" * 60)
print("History Aware RAG Chat")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

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

    print("\nAI :", result["answer"])