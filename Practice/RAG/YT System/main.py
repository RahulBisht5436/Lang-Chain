import sys
from pathlib import Path

import numpy as np

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2])
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from llm.openAI_llm import llm
from llm.openaiEmbedding import embeddings

from transScriptLoader import video_transcript_loader
from transcriptChunker import split_text
from generate_embedding import transScript_embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vectors, query_vector):

    vectors = np.array(vectors)
    query_vector = np.array(query_vector)

    similarity = np.dot(
        vectors,
        query_vector
    ) / (
        np.linalg.norm(
            vectors,
            axis=1
        )
        *
        np.linalg.norm(
            query_vector
        )
    )

    return similarity


# ============================================================
# GET TOP MATCHING CHUNKS
# ============================================================

def get_relevant_chunks(
    chunks,
    vectors,
    query_vector,
    top_k=4
):

    # Calculate similarity between
    # question and every transcript chunk

    similarity_scores = cosine_similarity(
        vectors,
        query_vector
    )

    # Get indices of highest similarity scores
    #
    # Example:
    #
    # [0.21, 0.82, 0.45, 0.91]
    #
    # top_k = 2
    #
    # [3, 1]

    top_indices = np.argsort(
        similarity_scores
    )[-top_k:][::-1]

    relevant_chunks = []

    for index in top_indices:

        relevant_chunks.append(
            {
                "index": index,
                "score": similarity_scores[index],
                "chunk": chunks[index]
            }
        )

    return relevant_chunks


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are an expert assistant.

Answer the question using ONLY the provided context.

Question:
{question}

Context:
{context}

Rules:

1. Use only the information provided in the context.
2. Do not use your own knowledge.
3. Do not make up information.
4. If the answer is not present in the context, respond:

Provide answer in English Language 

"Unable to give you an answer."
""",
    input_variables=[
        "question",
        "context"
    ]
)


# ============================================================
# OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# LANGCHAIN CHAIN
# ============================================================

chain = prompt | llm | parser


# ============================================================
# USER INPUT
# ============================================================

video_url = input(
    "Enter the video link: "
)

question = input(
    "Enter the Question: "
)


# ============================================================
# RAG PIPELINE
# ============================================================

if video_url and question:

    # ========================================================
    # STEP 1: LOAD YOUTUBE TRANSCRIPT
    # ========================================================

    raw_transcript = video_transcript_loader(
        video_url
    )

    print(
        "\nLoading Step completed"
        " ==============================>>>>>\n"
    )


    # ========================================================
    # STEP 2: SPLIT TRANSCRIPT INTO CHUNKS
    # ========================================================

    splitted_transScript = split_text(
        raw_transcript
    )

    print(
        "Splitting Completed"
        " ==========================>>>>>"
    )

    print(
        "Total chunks:",
        len(splitted_transScript)
    )


    # ========================================================
    # STEP 3: CREATE EMBEDDINGS FOR CHUNKS
    # ========================================================

    vectorsTransScript = transScript_embedding(
        splitted_transScript
    )

    print(
        "\nEmbedding completed"
        " ===================================>>>>>>"
    )


    # ========================================================
    # STEP 4: CREATE EMBEDDING FOR QUESTION
    # ========================================================

    vectorQuestion = embeddings.embed_query(
        question
    )

    print(
        "Question embedding completed"
        " ===============================>>>>>>"
    )


    # ========================================================
    # STEP 5: FIND RELEVANT CHUNKS
    # ========================================================

    relevant_chunks = get_relevant_chunks(
        chunks=splitted_transScript,
        vectors=vectorsTransScript,
        query_vector=vectorQuestion,
        top_k=4
    )

    print(
        "\nRelevant chunks retrieved"
        " ===============================>>>>>"
    )


    # ========================================================
    # STEP 6: DISPLAY MATCHING CHUNKS
    # ========================================================

    for result in relevant_chunks:

        print(
            "\n=========================================="
        )

        print(
            "Chunk Index:",
            result["index"]
        )

        print(
            "Similarity Score:",
            result["score"]
        )

        print(
            "=========================================="
        )

        print(
            result["chunk"]
        )


    # ========================================================
    # STEP 7: CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        result["chunk"]
        for result in relevant_chunks
    )

    print(
        "\nContext Generated"
        " ===================================>>>>>"
    )


    # ========================================================
    # STEP 8: SEND QUESTION + CONTEXT TO LLM
    # ========================================================

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )


    # ========================================================
    # STEP 9: FINAL ANSWER
    # ========================================================

    print(
        "\n\n=========================================="
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=========================================="
    )

    print(
        response
    )