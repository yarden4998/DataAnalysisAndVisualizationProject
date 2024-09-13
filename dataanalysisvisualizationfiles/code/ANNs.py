import faiss
from annoy import AnnoyIndex
import numpy as np

class ANN:
    def __init__(self, dimension, metric='L2', use_gpu=False, num_threads=2):
        self.dimension = dimension
        self.metric = metric
        if metric == 'L2':
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == 'InnerProduct':
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == 'Cosine':
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == 'LSH':
            self.index = faiss.IndexLSH(dimension, 8*dimension)
        elif metric == 'HNSW':
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        elif metric == 'IVF_l2':    
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100, faiss.METRIC_L2)
        elif metric == 'IVF_ip':    
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, 100, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Unsupported metric. Supported metrics are: L2, InnerProduct, Cosine, LSH, HNSW, IVF_l2, IVF_ip")
        
        if use_gpu:
            res = faiss.StandardGpuResources() 
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            faiss.omp_set_num_threads(num_threads)

    def add_embeddings(self, embeddings):
        if self.metric == 'Cosine' or self.metric == 'IVF_ip':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if self.metric == 'IVF_l2' or self.metric == 'IVF_ip':
            self.index.train(embeddings)
        self.index.add(embeddings)
    
    def compute_distance_matrix(self, embeddings):
        if self.metric == 'Cosine' or self.metric == 'IVF_ip':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        n = embeddings.shape[0]
        distance_matrix = np.zeros((n, n))
        index_matrix = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            D, I = self.index.search(embeddings[i:i+1], n)
            distance_matrix[i, :] = D[0]
            index_matrix[i, :] = I[0]
        return np.triu(distance_matrix), np.triu(index_matrix)
    
    def search(self, query, k):
        if self.metric == 'Cosine' or self.metric == 'IVF_ip':
            query = query / np.linalg.norm(query)
        D, I = self.index.search(query, k)
        return D, I

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

    def search(self, query_vectors, k):
        indices = [self.index.get_nns_by_vector(vector, k) for vector in query_vectors]
        distances = [[] for _ in range(len(query_vectors))]
        return distances, indices