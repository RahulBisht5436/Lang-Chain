from llm import ollamaEmbedding

ques = input("What is the prompt\n")

# inside the embedding model we use the embed_query
response = ollamaEmbedding.embeddings.embed_query(ques)

print(response)
print(len(response))   # Prints the vector dimension