import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
# Ensure nltk resources are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_list_target(genre_str: str) -> list[str]:
    """
    Preprocess the genre string by splitting it and removing leading/trailing whitespaces.
    param genre_str: The genre string to preprocess
    """
    if pd.isna(genre_str):
        return []
    return [genre.strip() for genre in genre_str.split(',')]

def preprocess_target(genre_str: str) -> str:
    """
    Preprocess the genre string by splitting it and removing leading/trailing whitespaces.
    param genre_str: The genre string to preprocess
    """
    if pd.isna(genre_str):
        return ''
    return genre_str.strip() 


def preprocess_text(description: str) -> str:
    """
    Preprocess the description.
    param description: The description to preprocess
    """
    if pd.isna(description):
        return ''
    words = re.findall(r'\b\w+\b', description.lower())
    words = [word.strip() for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


def preprocess_data(dataset_name: str = 'imdb') -> pd.DataFrame:
    # Loads the csv file into a DataFrame
    data_path = get_path(dataset_name) 
    dataset = pd.read_csv(data_path)

    # Remove rows with missing values in the text and target columns
    text_name, target_name = get_text_target_columns(dataset_name)
    dataset.dropna(subset=[text_name], inplace=True)
    dataset.dropna(subset=[target_name], inplace=True)

    if dataset_name == 'imdb' or dataset_name == 'academic_papers':
        dataset['genre'] = dataset[target_name].apply(preprocess_list_target) 
        # Number of labels per item
        dataset['num_genres'] = dataset['genre'].apply(len)
    
    else:
        dataset['genre'] = dataset[target_name].apply(preprocess_list_target)

    # Preprocess the text column
    dataset['description_processed'] = dataset[text_name].apply(preprocess_text)

    # keep only the columns we need
    dataset = dataset[['description_processed', 'genre']]

    return dataset

def get_path(dataset: str):
    # Gets path to dataset given its name
    if dataset == 'imdb':
        return '/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/IMDb movies.csv'
    elif dataset == 'academic_papers':
        return '/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/arxiv_data_210930-054931.csv'
    elif dataset == 'legal':
        return '/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/legal_text_classification.csv'
    elif dataset == 'reviews':
        return '/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/product_reviews_40k.csv'            
    elif dataset == 'ecommerce':
        return '/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/ecommerceDataset.csv'
    else:
        raise ValueError('Invalid dataset name')
    
def get_text_target_columns(dataset: str):
    # Gets column names in to dataset given dataset name
    if dataset == 'imdb':
        return 'description', 'genre'
    elif dataset == 'academic_papers':
        return 'abstracts', 'terms'
    elif dataset == 'legal':
        return 'case_text', 'case_outcome'
    elif dataset == 'reviews':
        return 'Text', 'Cat1'
    elif dataset == 'ecommerce':
        return 'description', 'category'
    else:
        raise ValueError('Invalid dataset name')


