import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
# Ensure nltk resources are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_genre(genre_str: str) -> list[str]:
    """
    Preprocess the genre string by splitting it and removing leading/trailing whitespaces.
    param genre_str: The genre string to preprocess
    """
    if pd.isna(genre_str):
        return []
    return [genre.strip() for genre in genre_str.split(',')]


def preprocess_description(description: str) -> str:
    """
    Preprocess the description.
    param description: The description to preprocess
    """
    if pd.isna(description):
        return ''
    words = re.findall(r'\b\w+\b', description.lower())
    words = [word.strip() for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


def preprocess_data(data_path: str = '../data/IMDb movies.csv') -> pd.DataFrame:
    # Loads the csv file into a DataFrame
    imdb_movies_data = pd.read_csv(data_path)

    # Remove rows with missing values in the 'description' and 'genre' columns
    imdb_movies_data.dropna(subset=['description'], inplace=True)
    imdb_movies_data.dropna(subset=['genre'], inplace=True)

    imdb_movies_data['genre'] = imdb_movies_data['genre'].apply(preprocess_genre)

    # Number of genres per movie
    imdb_movies_data['num_genres'] = imdb_movies_data['genre'].apply(len)

    # Preprocess the description
    imdb_movies_data['description_processed'] = imdb_movies_data['description'].apply(preprocess_description)

    # keep only the columns we need
    imdb_movies_data = imdb_movies_data[['index', 'description_processed', 'genre']]

    return imdb_movies_data

# Load the data
imdb_movies_data = preprocess_data()
print(imdb_movies_data.head())
print(imdb_movies_data.shape)
print(imdb_movies_data.columns)


