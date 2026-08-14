import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from llm.openaiEmbedding import embeddings
from llm.openAI_llm import llm

# =========================================================
# Cache configuration
# =========================================================

LOCAL_CACHE_ENABLED = True
GLOBAL_CACHE_ENABLED = True
L1_CACHE_TTL = int(os.getenv("L1_CACHE_TTL", "300"))
L2_CACHE_TTL = int(os.getenv("L2_CACHE_TTL", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# =========================================================
# L1 local cache (in-memory)
# =========================================================

_l1_cache: dict[str, tuple[float, dict]] = {}

# =========================================================
# L2 Redis client (lazy init)
# =========================================================

_redis_client = None
_redis_available = False
_redis_last_error: str | None = None


def redis_reset():
    global _redis_client, _redis_available, _redis_last_error
    _redis_client = None
    _redis_available = False
    _redis_last_error = None


def _get_redis_client():
    global _redis_client, _redis_available, _redis_last_error

    if not GLOBAL_CACHE_ENABLED:
        _redis_last_error = "L2 Redis cache is disabled"
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        _redis_client.ping()
        _redis_available = True
        _redis_last_error = None
        return _redis_client
    except Exception as exc:
        _redis_available = False
        _redis_client = None
        _redis_last_error = str(exc)
        return None


def get_redis_status() -> dict:
    client = _get_redis_client()
    if client is None:
        return {
            "connected": False,
            "error": _redis_last_error or "Could not connect to Redis",
            "url": REDIS_URL,
        }

    try:
        client.ping()
        return {
            "connected": True,
            "error": None,
            "url": REDIS_URL,
        }
    except Exception as exc:
        redis_reset()
        return {
            "connected": False,
            "error": str(exc),
            "url": REDIS_URL,
        }


def redis_health_check() -> bool:
    return get_redis_status()["connected"]


# =========================================================
# RAG chain (lazy init)
# =========================================================

_rag_chain = None
_vectorstore = None


def _get_rag_chain():
    global _rag_chain, _vectorstore

    if _rag_chain is not None:
        return _rag_chain

    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""
You are an assistant that answers questions using the
provided context.

Context:
{embedding_variable}

Question:
{query}

Provide an appropriate answer based on the provided context.
""",
        input_variables=["query", "embedding_variable"],
    )

    _vectorstore = Chroma(
        collection_name="langchain",
        embedding_function=embeddings,
        persist_directory=str(PROJECT_ROOT / "chroma_db_persist"),
    )

    _rag_chain = prompt | llm | parser
    return _rag_chain


def _make_cache_key(question: str) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _l1_get(cache_key: str) -> dict | None:
    if not LOCAL_CACHE_ENABLED:
        return None

    entry = _l1_cache.get(cache_key)
    if entry is None:
        return None

    expires_at, value = entry
    if time.time() > expires_at:
        _l1_cache.pop(cache_key, None)
        return None

    return value


def _l1_set(cache_key: str, value: dict):
    if not LOCAL_CACHE_ENABLED:
        return

    _l1_cache[cache_key] = (time.time() + L1_CACHE_TTL, value)


def _l2_get(cache_key: str) -> dict | None:
    client = _get_redis_client()
    if client is None:
        return None

    try:
        raw = client.get(f"rag:{cache_key}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def _l2_set(cache_key: str, value: dict):
    client = _get_redis_client()
    if client is None:
        return

    try:
        client.setex(
            f"rag:{cache_key}",
            L2_CACHE_TTL,
            json.dumps(value),
        )
    except Exception:
        pass


def _run_rag(question: str) -> dict:
    chain = _get_rag_chain()
    documents = _vectorstore.similarity_search(question)
    answer = chain.invoke(
        {
            "query": question,
            "embedding_variable": documents,
        }
    )

    return {
        "answer": answer,
        "documents": len(documents),
    }


def ask(question: str) -> dict:
    start = time.perf_counter()
    cache_key = _make_cache_key(question)

    cached = _l1_get(cache_key)
    if cached is not None:
        return {
            "source": "L1 Local Cache",
            "answer": cached["answer"],
            "time": time.perf_counter() - start,
            "documents": cached["documents"],
            "cache_key": cache_key,
        }

    cached = _l2_get(cache_key)
    if cached is not None:
        _l1_set(cache_key, cached)
        return {
            "source": "L2 Redis Cache",
            "answer": cached["answer"],
            "time": time.perf_counter() - start,
            "documents": cached["documents"],
            "cache_key": cache_key,
        }

    result = _run_rag(question)
    _l1_set(cache_key, result)
    _l2_set(cache_key, result)

    return {
        "source": "RAG + LLM",
        "answer": result["answer"],
        "time": time.perf_counter() - start,
        "documents": result["documents"],
        "cache_key": cache_key,
    }
