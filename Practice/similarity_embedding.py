from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# Create Embedding Model
# -------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------------
# In-Memory Knowledge Base
# -------------------------------------------------------

knowledge_base = [
    "Python is a programming language.",
    "LangChain is used for building LLM applications.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "MongoDB is a NoSQL database.",
    "React is a JavaScript library for building user interfaces.",
    "Node.js is used for backend development.",
    "The capital of India is New Delhi.",
    "Football is one of the most popular sports in the world."
]

# -------------------------------------------------------
# Generate embeddings for the stored data
# (Done only once)
# -------------------------------------------------------

knowledge_embeddings = embeddings.embed_documents(knowledge_base)

# -------------------------------------------------------
# User Input
# -------------------------------------------------------

query = input("Enter your query: ")

# -------------------------------------------------------
# Generate embedding for user query
# -------------------------------------------------------

query_embedding = embeddings.embed_query(query)

# -------------------------------------------------------
# Calculate Similarity
# -------------------------------------------------------

results = []

for sentence, embedding in zip(knowledge_base, knowledge_embeddings):

    similarity = cosine_similarity(
        [query_embedding],
        [embedding]
    )[0][0]

    results.append(
        {
            "text": sentence,
            "score": similarity
        }
    )

# -------------------------------------------------------
# Sort by similarity (Highest First)
# -------------------------------------------------------

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

# -------------------------------------------------------
# Print all results
# -------------------------------------------------------

print("\nSimilarity Results")
print("-" * 60)

for item in results:

    print(f"{item['score'] * 100:.2f}%  -->  {item['text']}")

# -------------------------------------------------------
# Print Best Match
# -------------------------------------------------------

best_match = results[0]

print("\n" + "=" * 60)
print("Most Similar Sentence")
print("=" * 60)

print(f"Sentence   : {best_match['text']}")
print(f"Similarity : {best_match['score'] * 100:.2f}%")