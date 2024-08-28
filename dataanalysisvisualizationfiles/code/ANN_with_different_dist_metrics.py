import faiss
import numpy as np

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
