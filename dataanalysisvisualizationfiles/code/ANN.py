import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocess_data import preprocess_data

class ANN:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_embeddings(self, embeddings):
        self.index.add(embeddings)
    
    def compute_distance_matrix(self, embeddings):
        n = embeddings.shape[0]
        distance_matrix = np.zeros((n, n))
        index_matrix = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            D, I = self.index.search(embeddings[i:i+1], n)
            distance_matrix[i, :] = D[0]
            index_matrix[i, :] = I[0]
        
        # Return the upper triangular matrix for distances and indices
        return np.triu(distance_matrix), np.triu(index_matrix)

# Load the dataset
df = preprocess_data()
df = df.head(10)

# Initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

print("Computing sentence embeddings...")
# Create sentence embeddings
sentences = df['description_processed'].tolist()
sentence_embeddings = model.encode(sentences)
print("Done!")

print("Computing distance matrix...")
# Initialize ANN class
ann = ANN(dimension=sentence_embeddings.shape[1])
print("Done!")

print("Adding embeddings to the index...")
# Add embeddings to the index
ann.add_embeddings(sentence_embeddings)
print("Done!")

print("Computing the distance matrix...")
# Compute the distance matrix
distance_matrix, index_matrix = ann.compute_distance_matrix(sentence_embeddings)
print("Done!")

# Display the distance matrix
print(distance_matrix)