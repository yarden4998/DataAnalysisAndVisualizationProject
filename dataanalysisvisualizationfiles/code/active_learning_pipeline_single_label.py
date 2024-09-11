import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from preprocess_data import preprocess_data
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import faiss

# Load your data
data = preprocess_data()

data['num_genres'] = data['genre'].apply(len)
data = data[data['num_genres'] == 1]

# Assuming data is a pandas DataFrame with 'description_processed' and 'genre' columns
descriptions = data['description_processed'].tolist()
genres = data['genre'].tolist()

# Encode the genres as numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(genres)

# Load a pre-trained sentence-transformer model to convert text to embeddings
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('bert-base-nli-mean-tokens')


# Convert descriptions to vector embeddings
X = model.encode(descriptions, show_progress_bar=True)

# Split the data into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Active Learning parameters
initial_train_size = 50  # Initial training set size
iterations = 38  # Number of iterations for active learning
sample_size = 500  # Samples to add per iteration

# Start by selecting the furthest points from each other for initial training
np.random.seed(42)
pool_indices = np.random.choice(len(X_train_full), initial_train_size, replace=False)
X_train = X_train_full[pool_indices]
y_train = y_train_full[pool_indices]

# Initialize FAISS Index for ANN search
# res = faiss.StandardGpuResources() 
faiss.omp_set_num_threads(12)
embedding_dim = X_train.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(X_train)  # Add only the training data to the index
# index = faiss.index_cpu_to_gpu(res, 0, index)

# Remaining pool of indices
remaining_indices = list(set(range(len(X_train_full))) - set(pool_indices))



# Initialize classifier for single-label classification
clf = RandomForestClassifier()

# Active Learning Loop using FAISS for selecting distant points
for iteration in range(iterations):
    # Train the classifier on the current training set
    clf.fit(X_train, y_train)
    
    # Optionally evaluate the classifier on the test set after each iteration (can be commented out)
    y_pred = clf.predict(X_test)
    print(f"Iteration {iteration + 1}: Test Accuracy = {accuracy_score(y_test, y_pred):.4f}")

    # Stop if there are not enough remaining samples or if the max training set size is reached
    if len(remaining_indices) == 0 or len(remaining_indices) < sample_size or len(X_train) >= len(X_train_full):
        break

    # Search for the most distant points from the current training set
    D_pool, _ = index.search(X_train_full[remaining_indices], len(X_train))
    
    # Compute the mean distance to the current training set for all remaining points
    avg_distances = D_pool.mean(axis=1)

    # Select the indices of the most distant points in the remaining pool
    furthest_indices_within_remaining = np.argsort(avg_distances)[-sample_size:]

    # Map the selected indices back to the original dataset indices
    selected_indices = [remaining_indices[i] for i in furthest_indices_within_remaining]

    # Add the selected samples to the training set
    X_train = np.vstack([X_train, X_train_full[selected_indices]])
    y_train = np.concatenate([y_train, y_train_full[selected_indices]], axis=0)

    # Remove the selected indices from the pool
    remaining_indices = list(set(remaining_indices) - set(selected_indices))
    
    index.add(X_train_full[selected_indices])


# Final evaluation on the test set
y_pred_final = clf.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final Accuracy: {final_accuracy:.4f}")

# Assuming y is your true labels and y_pred_final is the predicted labels
print(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))
