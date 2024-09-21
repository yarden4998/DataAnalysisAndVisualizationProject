# Addressing Class Imbalance in Active Learning via Approximate Nearest Neighbour Methods 

## Project Overview

This project focuses on addressing class imbalance in Active Learning pipelines using Approximate Nearest Neighbor (ANN) methods. The research investigates whether ANN can improve the efficiency and accuracy of Active Learning models when dealing with imbalanced, high-dimensional datasets.

The proposed approach includes exploring different indexing and sampling strategies, such as distance-based and cluster-based methods, to optimize data selection in active learning pipelines.

## Setup Instructions

### Prerequisites

* Python 3.10
* Poetry

### Environment Setup

Follow these steps to set up the environment and run the code:

1. **Install Poetry** \
    Ensure Poetry is installed on your machine. If not, install it by running:
    ```bash
    pip install poetry
    ```
2. **Select Poetry Base Interpreter** 
    ```bash
    poetry env use <path to your executable python 3.10>
    ```
3. **Check Poetry Base Interpreter** 
    Run the following command and check that the base interpreter is Python 3.10:
    ```bash
    poetry env info
    ```
4. **Clone This Repo** 
    ```bash
    git clone https://github.com/yarden4998/DataAnalysisAndVisualizationProject.git
    ```
5. **Run from Main Root**
    Navigate to the directory where the `pyproject.toml` file is located and run:
    ```bash
    poetry install --no-root
    ```
6. **Select Your Virtual Environment**
    Your virtual environment is now located at:
    ```
    .cache/pypoetry/virtualenvs
    ```

This will install all necessary dependencies for the project. Make sure to use the virtual environment created by Poetry.

## Running the Code

Please go to the [graphing & combinations pipeline notebook](dataanalysisvisualizationfiles/code/graphing%20%26%20combinations%20pipeline.ipynb) and select the desired configuration to run:
1. In cell number 2, please select the dataset.
2. In cell number 4, please select the methods.
3. Select the Poetry virtual environment.
4. Run the notebook.

To run any other code file, please use the created virtual environment.

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
│   │   ├── results/ # Directory for storing results of AL ANN running on different settings
│   ├── research/ # Directory for research-related files
│   │   ├── code/ # Directory for research-related code
│   │   │   ├── explore_datasets.ipynb # Jupyter notebook for exploring datasets
│   │   ├── data/ # Directory for research-related data
│   │   │   ├── IMDB Dataset.csv # IMDB dataset
│   │   │   ├── IMDb movies.csv # IMDb movies dataset
│   │   │   ├── Skin cancer ISIC/ # Skin cancer dataset
```

<!-- ## Conclusions -->

<!-- here is how to put here an image: ![image]('/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/images/academic_papers_accuracy_active_learning.png')

here is how to put here a link to anything (can add link to internet or to somewhere in the proj) : [name to appear for link]('/home/student/idan/DataAnalysisAndVisualizationProject/dataanalysisvisualizationfiles/data/images/academic_papers_accuracy_active_learning.png') -->

