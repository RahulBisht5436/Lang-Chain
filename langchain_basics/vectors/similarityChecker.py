from llm import ollamaEmbedding
import numpy as np

input1 = input("What is the input first \n")
input2 = input("What is the input second \n")

# inside the embedding model we use the embed_query
response1 = ollamaEmbedding.embeddings.embed_query(input1)
response2= ollamaEmbedding.embeddings.embed_query(input2)


print(np.dot(response1,response2))