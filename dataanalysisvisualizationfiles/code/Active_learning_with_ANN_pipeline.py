import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess_data import preprocess_data
import pandas as pd

np.random.seed(42)

def get_encoded_data(model_name: str, dataset_name: str) -> tuple[list, list, LabelEncoder]:
    """
    Load and preprocess data, then encode descriptions into embeddings and genres into numerical labels.

    Parameters:
    - model_name (str): The name of the pre-trained sentence-transformer model.
    - dataset_name (str): The name of the dataset to load.

    Returns:
    - X (list): List of encoded descriptions.
    - y (list): List of encoded genres.
    - label_encoder (LabelEncoder): The label encoder used for encoding genres.
    """
    data = preprocess_data(dataset_name)

    data['num_genres'] = data['genre'].apply(len)
    data = data[data['num_genres'] == 1]

    descriptions = data['description_processed'].tolist()
    genres = data['genre'].tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(genres)

    model = SentenceTransformer(model_name)
    X = model.encode(descriptions, show_progress_bar=True)

    return X, y, label_encoder

def find_distant_points(distance_method, D_pool, sample_size):
    """
    Find the most distant points in the pool based on the specified distance method.

    Parameters:
    - distance_method (str): The method to compute distances ('min', 'max', 'avg').
    - D_pool (np.ndarray): Distance matrix.
    - sample_size (int): Number of samples to select.

    Returns:
    - furthest_indices_within_remaining (np.ndarray): Indices of the most distant points.
    """
    if distance_method == 'min':
        distances = D_pool.min(axis=1)
    elif distance_method == 'max':
        distances = D_pool.max(axis=1)
    else:
        distances = D_pool.mean(axis=1)

    furthest_indices_within_remaining = np.argsort(distances)[-sample_size:]

    return furthest_indices_within_remaining

def run_pipeline(clf, iterations, sample_size, initial_train_size, label_encoder, X, y, distance_method='avg', use_faiss_clustering=False, ensure_genre_coverage=False, use_ann_selection=True):
    """
    Run the active learning pipeline with the specified parameters.

    Parameters:
    - clf: The classifier to use.
    - iterations (int): Number of iterations for active learning.
    - sample_size (int): Number of samples to add per iteration.
    - initial_train_size (int): Initial training set size.
    - label_encoder (LabelEncoder): The label encoder used for encoding genres.
    - X (list): List of encoded descriptions.
    - y (list): List of encoded genres.
    - distance_method (str): The method to compute distances ('min', 'max', 'avg').
    - use_faiss_clustering (bool): Whether to use FAISS clustering for initial sample selection.
    - ensure_genre_coverage (bool): Whether to ensure at least one example from each genre in the initial training set.
    - use_ann_selection (bool): Whether to use ANN for selecting distant points.

    Returns:
    - final_accuracy (float): Final accuracy on the test set.
    - report (str): Classification report.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pool_indices = []

    if ensure_genre_coverage:
        unique_genres = np.unique(y_train_full)
        for genre in unique_genres:
            genre_indices = np.where(y_train_full == genre)[0]
            selected_index = np.random.choice(genre_indices)
            pool_indices.append(selected_index)

    if use_faiss_clustering:
        remaining_indices = list(set(range(len(X_train_full))) - set(pool_indices))
        remaining_data = X_train_full[remaining_indices]

        n_clusters = initial_train_size - len(pool_indices)
        kmeans = faiss.Kmeans(d=remaining_data.shape[1], k=n_clusters, niter=20, verbose=True)
        kmeans.train(remaining_data)
        cluster_labels = kmeans.index.search(remaining_data, 1)[1].flatten()

        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            selected_index = remaining_indices[np.random.choice(cluster_indices)]
            pool_indices.append(selected_index)
    else:
        remaining_indices = list(set(range(len(X_train_full))) - set(pool_indices))
        additional_indices = np.random.choice(remaining_indices, initial_train_size - len(pool_indices), replace=False)
        pool_indices.extend(additional_indices)

    X_train = X_train_full[pool_indices]
    y_train = y_train_full[pool_indices]

    faiss.omp_set_num_threads(12)
    embedding_dim = X_train.shape[1]
    # index = faiss.IndexFlatIP(embedding_dim)
    nbits = 16  # Number of bits for hashing
    index = faiss.IndexLSH(embedding_dim, nbits)
    index.add(X_train)

    remaining_indices = list(set(range(len(X_train_full))) - set(pool_indices))

    for iteration in range(iterations):
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print(f"Iteration {iteration + 1}: Test Accuracy = {accuracy_score(y_test, y_pred):.4f}")

        if len(remaining_indices) == 0 or len(remaining_indices) < sample_size or len(X_train) >= len(X_train_full):
            break

        if use_ann_selection:
            D_pool, _ = index.search(X_train_full[remaining_indices], len(X_train))
            furthest_indices_within_remaining = find_distant_points(distance_method, D_pool, sample_size)
        else:
            furthest_indices_within_remaining = np.random.choice(len(remaining_indices), sample_size, replace=False)

        selected_indices = [remaining_indices[i] for i in furthest_indices_within_remaining]

        X_train = np.vstack([X_train, X_train_full[selected_indices]])
        y_train = np.concatenate([y_train, y_train_full[selected_indices]], axis=0)

        remaining_indices = list(set(remaining_indices) - set(selected_indices))
        
        index.add(X_train_full[selected_indices])

    y_pred_final = clf.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    print(f"Final Accuracy: {final_accuracy:.4f}")

    report = classification_report(y_test, y_pred_final, target_names=label_encoder.classes_)
    print(report)

    return final_accuracy, report

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
clf = SGDClassifier()

for dataset in ['imdb', 'academic_papers', 'legal', 'reviews', 'ecommerce']:
    X, y, label_encoder = get_encoded_data(MODEL_NAME, dataset)

    initial_train_size = 250
    iterations = 20
    sample_size = 500

    results = []

    # Select initialization type
    for use_faiss_clustering in [True, False]:
        #select whether ot initialy select at least one example from each genre
        for ensure_genre_coverage in [True, False]:
            # Select whether to use ANN for selecting distant points or to randomally select them
            for use_ann_selection in [True, False]:
                print(f"Running with use_faiss_clustering={use_faiss_clustering}, ensure_genre_coverage={ensure_genre_coverage}, use_ann_selection={use_ann_selection}")
                final_accuracy, report = run_pipeline(clf, iterations, sample_size, initial_train_size, label_encoder, X, y, 'avg', use_faiss_clustering, ensure_genre_coverage, use_ann_selection)
                results.append((use_faiss_clustering, ensure_genre_coverage, use_ann_selection, final_accuracy, report))
                
                # Append results to the file after each iteration
                with open(f"dataanalysisvisualizationfiles/data/results/AL_ANN_results_faissLSH_{dataset}.txt", "a") as f:
                    f.write(f"use_faiss_clustering={use_faiss_clustering}, ensure_genre_coverage={ensure_genre_coverage}, use_ann_selection={use_ann_selection}\n")
                    f.write(f"Final Accuracy: {final_accuracy:.4f}\n")
                    f.write(report)
                    f.write("\n\n")