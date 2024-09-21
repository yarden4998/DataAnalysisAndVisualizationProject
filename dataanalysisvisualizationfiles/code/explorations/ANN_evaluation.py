import os
import sys
import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from ANNs import ANN, ANNOY
from time import time
from sklearn.metrics import pairwise_distances_argmin_min
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Add the directory containing preprocess_data.py to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(module_path)

from preprocess_data import preprocess_data

def load_data():
    df = preprocess_data()
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    print("Computing sentence embeddings...")
    sentences = df['description_processed'].tolist()
    sentence_embeddings = model.encode(sentences)
    print("Done!")
    return sentence_embeddings

def recall_at_k(true_neighbors, predicted_neighbors, k):
    recall = []
    for true, pred in zip(true_neighbors, predicted_neighbors):
        recall.append(len(set(true) & set(pred)) / len(true))
    return np.mean(recall)

def compute_true_neighbors(vectors, query_vectors, k):
    distances = cdist(query_vectors, vectors, metric='cosine')
    true_neighbors = np.argsort(distances, axis=1)[:, :k]
    return true_neighbors

def evaluate_index(index, vectors, query_vectors, k_values, true_neighbors):
    index.add_embeddings(vectors)
    times, recalls = [], []
    for k in k_values:
        start_time = time()
        _, indices = index.search(query_vectors, k)
        elapsed_time = time() - start_time
        predicted_neighbors = indices
        recall = recall_at_k(true_neighbors, predicted_neighbors, k)
        times.append(elapsed_time)
        recalls.append(recall)
    return times, recalls

def main():
    k_values = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

    vectors = load_data()
    query_vectors = np.random.rand(2000, vectors.shape[1]).astype(np.float32)
    true_neighbors = compute_true_neighbors(vectors, query_vectors, k_values[-1])

    f_l2 = ANN(dimension=vectors.shape[1], metric='L2', use_gpu=True)
    f_cos = ANN(dimension=vectors.shape[1], metric='Cosine', use_gpu=True)
    f_ip = ANN(dimension=vectors.shape[1], metric='InnerProduct', use_gpu=True)
    f_cos_nogpu_multi = ANN(dimension=vectors.shape[1], metric='Cosine', use_gpu=False, num_threads=12)
    f_cos_nogpu = ANN(dimension=vectors.shape[1], metric='Cosine', use_gpu=False)
    f_lsh = ANN(dimension=vectors.shape[1], metric='LSH', use_gpu=True)
    f_hnsw = ANN(dimension=vectors.shape[1], metric='HNSW', use_gpu=True)
    f_ivf_l2 = ANN(dimension=vectors.shape[1], metric='IVF_l2', use_gpu=False, num_threads=12)
    f_ivf_cosine = ANN(dimension=vectors.shape[1], metric='IVF_ip', use_gpu=False, num_threads=12)
    an_cos = ANNOY(dimension=vectors.shape[1], metric='angular', n_trees=10)
    an_l2 = ANNOY(dimension=vectors.shape[1], metric='euclidean', n_trees=10)
    an_ip = ANNOY(dimension=vectors.shape[1], metric='dot', n_trees=10)
    an_cos_100 = ANNOY(dimension=vectors.shape[1], metric='angular', n_trees=100)
    indices = {'faiss_l2': f_l2, 'faiss_inner_product': f_ip, 'faiss_cosine':f_cos, 'faiss_cosine_no_gpu_multi': f_cos_nogpu_multi, 'faiss_cosine_no_gpu_no_multi':f_cos_nogpu, 'faiss_lsh':f_lsh, 'faiss_HNSW':f_hnsw, 'faiss_IVF_L2':f_ivf_l2, 'faiss_IVF_Cosine':f_ivf_cosine, 'annoy_cosine':an_cos, 'annoy_l2':an_l2, 'annoy_dot_product':an_ip, 'annoy_cosine_100_trees':an_cos_100}

    times, recalls = {}, {}
    for name, index in indices.items():
        print(f"Evaluating {name}...")
        t, r = evaluate_index(index, vectors, query_vectors, k_values, true_neighbors)
        times[name] = t
        recalls[name] = r
        print(f"Done with {name}!")

    print(pd.DataFrame(recalls, index=k_values))
    plt.figure(figsize=(12, 8))
    for key, values in recalls.items():
        plt.plot(k_values, values, label=key)    
    plt.xlabel('K')
    plt.ylabel('Recall')
    plt.title('Recall@k for Different Indices')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ann_recall.png')
    plt.show()

    print(pd.DataFrame(times, index=k_values))
    plt.figure(figsize=(12, 8))
    for key, values in times.items():
        plt.plot(k_values, values, label=key)    
    plt.xlabel('K')
    plt.ylabel('Time')
    plt.title('Runtime@k for Different Indices')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ann_runtime.png')
    plt.show()


if __name__ == "__main__":
    main()