# Addressing Class Imbalance in Active Learning via Approximate Nearest Neighbour Methods 

## Project Overview

This project focuses on addressing class imbalance in Active Learning pipelines using Approximate Nearest Neighbor (ANN) methods. The research investigates wether ANN can improve the efficiency and accuracy of Active Learning models when dealing with imbalanced, high-dimensional datasets.

The proposed approach includes exploring different indexing and sampling strategies, such as distance-based and cluster-based methods, to optimize data selection in active learning pipelines.

## Setup Instructions

Follow these steps to set up the environment and run the code:

1. **Install Poetry** \
    Ensure Poetry is installed on your machine. If not, install it by running:
    ```bash
    pip install poetry
    ```
2. **Clone this repo** 
    ```bash
    git clone <repository_url>
    ```
3. **Run from main root**
    ```bash
    poetry install --no-root
    ```
This will install all necessary dependencies for the project. Make sure to use the virtual environment created by Poetry.

## Running the Code
<!-- 
1. **Preprocess the Data** \
Run the preprocessing script to clean the dataset, remove unnecessary values, and embed the text using a transformer-based model:

bash
poetry run python scripts/preprocessing.py

2. Run the Active Learning Pipeline
To run the Active Learning process, use the following command. This will perform sampling and model training based on the ANN methods discussed in the project.

bash
poetry run python scripts/active_learning_pipeline.py

3. Run Experiments and Generate Results
To run the ANN experiments and gather results such as recall@k and runtime, execute the experiment script:

bash
poetry run python scripts/experiments.py

4. Visualize the Results
Finally, to visualize the results of the experiments (e.g., using tables, plots), run:

bash
poetry run python scripts/visualization.py -->

## Project Structure

Below is an outline of the project files and directories:

```bash
DataAnalysisAndVisualizationProject/
├── pyproject.toml # Poetry configuration file for dependencies
├── README.md # This README file
├── poetry.lock # Poetry lock file for dependencies
├── dataanalysisvisualizationfiles/ # Main project directory
│   ├── code/ # Python scripts and notebooks for the project
│   │   ├── ANN_as_clustering.ipynb # Notebook for clustering using ANN
│   │   ├── graphing & combinations pipeline.ipynb # Notebook for graphing and combining pipelines
│   │   ├── Active_learning_with_ANN_pipeline.py # Script for active learning with ANN
│   │   ├── ANNs.py # Implementation of ANN models
│   │   ├── preprocess_data.py # Data preprocessing functions
│   │   ├── explorations/ # Exploratory notebooks and scripts
│   │   │   ├── full_pipeline_single_label.ipynb # Full pipeline for single label classification
│   │   │   ├── multilabel_pipeline.ipynb # Pipeline for multilabel classification
│   │   │   ├── ANN_evaluation.py # Evaluation of ANN models
│   │   │   ├── active_learning_pipeline_single_label.py # Active learning for single label classification
│   │   │   ├── AL_Pipeline_Baselines.ipynb # Baseline comparisons for active learning pipeline
│   │   │   ├── ANN_clustering_comparison.ipynb # Comparison of clustering methods using ANN
│   ├── data/ # Contains datasets used in this project
│   │   ├── ecommerceDataset.csv # E-commerce dataset
│   │   ├── product_reviews_40k.csv # Product reviews dataset
│   │   ├── IMDb movies.csv # IMDb movies dataset
│   │   ├── arxiv_data_210930-054931.csv # Arxiv data
│   │   ├── legal_text_classification.csv # Legal text classification dataset
│   │   ├── images/ # Directory of images for the final report
│   │   ├── pickle_files/ # Directory for saving pickle files
│   ├── research/ # Directory for research-related files
│   │   ├── code/ # Directory for research-related code
│   │   │   ├── explore_datasets.ipynb # Jupyter notebook for exploring datasets
│   │   ├── data/ # Directory for research-related data
│   │   │   ├── IMDB Dataset.csv # IMDB dataset
│   │   │   ├── IMDb movies.csv # IMDb movies dataset
│   │   │   ├── Skin cancer ISIC/ # Skin cancer dataset
```

## Conclusions

<!-- here is how to put here an image: ![image]('/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/images/academic_papers_accuracy_active_learning.png')

here is how to put here a link to anything (can add link to internet or to somewhere in the proj) : [name to appear for link]('/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/images/academic_papers_accuracy_active_learning.png') -->

