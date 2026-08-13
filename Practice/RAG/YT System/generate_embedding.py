import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from llm.openaiEmbedding import embeddings


def transScript_embedding(chunks):
    vectors = embeddings.embed_documents(chunks)

    print("Number of chunks:", len(chunks))
    print("Number of vectors:", len(vectors))

    print("First vector:")
    print(vectors[0])
    
    return vectors