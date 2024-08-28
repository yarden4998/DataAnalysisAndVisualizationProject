import faiss
import numpy as np
import pandas as pd
import
from sentence_transformers import SentenceTransformer
from preprocess_data import preprocess_data

class ANN:
    def __init__(self, dimension, metric='L2'):
        self.dimension = dimension
        
        if metric == 'L2':
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == 'InnerProduct':
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == 'Cosine':
            # Faiss does not have a direct cosine distance index, but we can simulate it
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("Unsupported metric. Choose from 'L2', 'InnerProduct', or 'Cosine'.")

    def add_embeddings(self, embeddings):
        if hasattr(self, 'metric') and self.metric == 'Cosine':
            # Normalize the embeddings if using cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings)
    
    def compute_distance_matrix(self, embeddings):
        if hasattr(self, 'metric') and self.metric == 'Cosine':
            # Normalize the embeddings if using cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

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

# List of metrics to compute
metrics = ['L2', 'InnerProduct', 'Cosine']

for metric in metrics:
    print(f"Computing distance matrix for {metric} similarity...")
    # Initialize ANN class with the current metric
    ann = ANN(dimension=sentence_embeddings.shape[1], metric=metric)
    print(f"ANN class with {metric} similarity Done!")

    print("Adding embeddings to the index...")
    # Add embeddings to the index
    ann.add_embeddings(sentence_embeddings)
    print("Done!")

    print(f"Computing the distance matrix for {metric} similarity...")
    # Compute the distance matrix
    distance_matrix, index_matrix = ann.compute_distance_matrix(sentence_embeddings)
    print("Done!")
    
    # Save distance_matrix and index_matrix to pkl files
    with open(f'distance_matrix_{metric}.pkl', 'wb') as f:
        pickle.dump(distance_matrix, f)
    with open(f'index_matrix_{metric}.pkl', 'wb') as f:
        pickle.dump(index_matrix, f)

    print(f"Saved {metric} similarity distance matrix and index matrix to pkl files.\n")
