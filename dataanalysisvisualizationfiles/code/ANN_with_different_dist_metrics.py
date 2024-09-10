import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from preprocess_data import preprocess_data
from time import time
from sklearn.metrics import pairwise_distances_argmin_min
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

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
        res = faiss.StandardGpuResources() 
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        

        
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

class ANNOY:

    def __init__(self, dimension, metric='angular', n_trees=10):
        self.dimension = dimension
        self.metric = metric
        self.index = AnnoyIndex(dimension, metric)
        self.n_trees = n_trees
    
    def add_embeddings(self, embeddings):
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)
        self.index.build(self.n_trees)
    
    def compute_distance_matrix(self, embeddings):
        n = len(embeddings)
        distance_matrix = np.zeros((n, n))
        index_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            I = self.index.get_nns_by_item(i, n, include_distances=True)
            index_matrix[i, :len(I[0])] = I[0]
            distance_matrix[i, :len(I[1])] = I[1]
        
        return np.triu(distance_matrix), np.triu(index_matrix)

# Load the dataset
df = preprocess_data()
#df = df.head(10)

# Initialize sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("Computing sentence embeddings...")
# Create sentence embeddings
sentences = df['description_processed'].tolist()
sentence_embeddings = model.encode(sentences)
print(len(df))
print("Done!")

# List of metrics to compute
k_values = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
metrics = ['L2', 'InnerProduct', 'Cosine', 'Annoy']
n_trees = 10
runtime_results = {}
recall_results = {}

exact_ann = ANN(dimension=sentence_embeddings.shape[1], metric='L2')
exact_ann.add_embeddings(sentence_embeddings)
_, exact_indices = exact_ann.compute_distance_matrix(sentence_embeddings)

for metric in metrics:
    print(f"Computing distance matrix for {metric} similarity...")

    if metric == 'Annoy':
        ann = ANNOY(dimension=sentence_embeddings.shape[1], metric='angular')
    else:
        ann = ANN(dimension=sentence_embeddings.shape[1], metric=metric)

    ann.add_embeddings(sentence_embeddings)
    start_time = time()
    distance_matrix, index_matrix = ann.compute_distance_matrix(sentence_embeddings)
    elapsed_time = time() - start_time
    runtime_results[metric] = elapsed_time
    recall_results[metric] = []

    for k in k_values:
        recall_at_k = np.mean([
            len(set(index_matrix[i, :k]) & set(exact_indices[i, :k])) / k
            for i in range(len(sentences))
        ])
        recall_results[metric].append(recall_at_k)

    print(f"Done! Time taken: {elapsed_time:.4f} seconds for {metric}.\n")

    with open(f'distance_matrix_{metric}.pkl', 'wb') as f:
        pickle.dump(distance_matrix, f)
    with open(f'index_matrix_{metric}.pkl', 'wb') as f:
        pickle.dump(index_matrix, f)

    print(f"Saved {metric} similarity distance matrix and index matrix to pkl files.\n")

plt.figure(figsize=(12, 6))

# Plot Recall@k
plt.subplot(1, 2, 1)
for metric in metrics:
    plt.plot(k_values, recall_results[metric], label=metric)
plt.title('Recall@k for Different Metrics')
plt.xlabel('k')
plt.ylabel('Recall@k')
plt.legend()
plt.grid(True)

# Plot Runtime
plt.subplot(1, 2, 2)
plt.bar(runtime_results.keys(), runtime_results.values())
plt.title('Runtime for Different Metrics')
plt.xlabel('Metric')
plt.ylabel('Time (seconds)')
plt.grid(True)

# Save the plots
plt.tight_layout()
plt.savefig('ann_recall_runtime.png')
plt.show()

print("Graphs saved as 'ann_recall_runtime.png'.")