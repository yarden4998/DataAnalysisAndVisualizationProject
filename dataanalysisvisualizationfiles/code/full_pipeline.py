import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from preprocess_data import preprocess_data
# Precompute the pairwise distance matrix (upper triangular)
from scipy.spatial.distance import pdist, squareform

# Load your data
data = preprocess_data()

# Assuming data is a pandas DataFrame with 'description_processed' and 'genre' columns
descriptions = data['description_processed'].tolist()
genres = data['genre'].tolist()

# Encode the genres as numerical labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(genres)

# Load a pre-trained sentence-transformer model to convert text to embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Convert descriptions to vector embeddings
X = model.encode(descriptions, show_progress_bar=True)

# Initialize FAISS Index for ANN search
embedding_dim = X.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(X)  # Adding all vectors to the index

# Active Learning parameters
initial_train_size = 50  # Initial training set size
iterations = 10  # Number of iterations for active learning
sample_size = 10  # Samples to add per iteration

# Start by selecting the furthest points from each other for initial training
np.random.seed(42)
initial_indices = np.random.choice(len(X), initial_train_size, replace=False)
X_train = X[initial_indices]
y_train = y[initial_indices]

# Remaining pool of indices
remaining_indices = list(set(range(len(X))) - set(initial_indices))

# Compute pairwise distances between all vectors
distance_matrix = squareform(pdist(X, metric='euclidean'))

clf = MultiOutputClassifier(RandomForestClassifier())

# Active Learning Loop with precomputed distances
for iteration in range(iterations):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X)
    print(f"Iteration {iteration + 1}: Accuracy = {accuracy_score(y, y_pred):.4f}")
    
    # Compute distances between the current training set and the rest of the dataset using precomputed matrix
    D = distance_matrix[initial_indices, :]  # Use the indices from the current training set
    
    # Select the furthest points from the training set
    furthest_indices = np.argsort(D.mean(axis=0))[-sample_size:]
    
    # Add the selected samples to the training set
    X_train = np.vstack([X_train, X[furthest_indices]])
    y_train = np.concatenate([y_train, y[furthest_indices]], axis=0)
    
    # Remove the selected indices from the pool
    remaining_indices = list(set(remaining_indices) - set(furthest_indices))
    
    if len(remaining_indices) == 0 or len(remaining_indices) < sample_size or len(X_train) >= 600:
        break

# # Active Learning Loop
# for iteration in range(iterations):
#     # Train a classifier (RandomForest for simplicity, but you can use any classifier)
#     clf = RandomForestClassifier()
#     clf.fit(X_train, y_train)
    
#     # Evaluate the classifier on the remaining pool (here you can split a validation set)
#     y_pred = clf.predict(X)
#     print(f"Iteration {iteration + 1}: Accuracy = {accuracy_score(y, y_pred):.4f}")
    
#     # Find the furthest points from the current training set
#     D, I = index.search(X_train, len(X))  # Compute distances of the remaining points to the training set
    
#     # Select the furthest neighbors
#     furthest_indices = np.argsort(D.mean(axis=1))[-sample_size:]
#     selected_indices = [remaining_indices[i] for i in furthest_indices]
    
#     # Add the selected samples to the training set
#     X_train = np.vstack([X_train, X[selected_indices]])
#     y_train = np.concatenate([y_train, y[selected_indices]], axis=0)
    
#     # Remove the selected indices from the pool
#     remaining_indices = list(set(remaining_indices) - set(selected_indices))

#     # If the pool is exhausted, break the loop
#     if len(remaining_indices) == 0:
#         break

# Final evaluation
y_pred_final = clf.predict(X)
final_accuracy = accuracy_score(y, y_pred_final)
print(f"Final Accuracy: {final_accuracy:.4f}")
