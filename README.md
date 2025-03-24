# Clustering Exploration

## Author: Michelangelo Zampieri

## Purpose
The objective of this project is to gain a deeper understanding of projection and clustering techniques. The dataset used for this exploration was sourced from Kaggle and was selected due to its relatively clean structure and ease of use. 

## The Data
The dataset is an income survey of Canadian citizens (or potentially citizens of another country), containing extensive demographic and financial information. Detailed descriptions of the dataset's features can be found in `./data/README`.

## Exploration
The initial exploration was conducted in `exploration.ipynb`. This notebook includes an overview of the dataset, the creation of histograms, and data preprocessing. Following this, various data projections were performed. Subsequently, clustering was conducted using KMeans and DBSCAN algorithms.

### Pre-processing
The dataset contains several categorical features, including `'Province', 'MBMREGP', 'Gender', 'Age_gap', 'Marital_status', 'Highschool', 'Highest_edu', 'Work_ref', 'Immigrant'`. These features were one-hot encoded, increasing the dimensionality of the dataset from 37 to 108. Additionally, `StandardScaler` was applied to normalize the data, addressing inconsistencies in feature scaling.

### Projections
Two projection techniques were utilized: Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) with a perplexity value of 50. These projections provided a clearer visualization of the data, facilitating the clustering process.

### Clustering
Two clustering algorithms, KMeans and DBSCAN, were applied to the dataset. Each algorithm was used on both PCA and t-SNE projections, resulting in a total of four clustering models. Hyperparameter tuning was performed for each model, exploring different cluster numbers for KMeans and varying `min_samples` values for DBSCAN. This process generated approximately 40 visualizations.

### Evaluation
The clustering models were evaluated using two metrics:
1. **Inertia**: Used to assess the compactness of clusters, with the elbow method applied to determine optimal hyperparameters.
2. **Silhouette Score**: Used to measure the quality of clustering by evaluating how similar an object is to its own cluster compared to other clusters.

Both metrics provided consistent results, aiding in the identification of optimal hyperparameters.

## Figures

The `figures` directory is organized into three main categories, each containing visualizations generated during the exploration and clustering processes:

1. **Exploration**:  
   This category includes initial exploratory visualizations, such as histograms of the dataset's features, as well as the first PCA and t-SNE projections. These plots provide an overview of the data distribution and its structure before clustering.

2. **KMEANS**:  
   This category is further divided into subdirectories:  
   - **KMEANS-PCA**: Contains visualizations of KMeans clustering applied to PCA-projected data. This includes cluster plots and evaluation graphs, such as inertia and silhouette score plots, which were used to assess the quality of clustering.  
   - **KMEANS-tSNE**: Contains visualizations of KMeans clustering applied to t-SNE-projected data. Similar to the PCA subdirectory, it includes cluster plots and evaluation graphs for hyperparameter tuning and model assessment.

3. **DBSCAN**:  
   This category is also divided into subdirectories:  
   - **DBSCAN-PCA**: Contains visualizations of DBSCAN clustering applied to PCA-projected data. This includes cluster plots and evaluation graphs, such as silhouette score plots, to analyze the performance of the clustering.  
   - **DBSCAN-tSNE**: Contains visualizations of DBSCAN clustering applied to t-SNE-projected data. These plots provide insights into the clustering results and the impact of hyperparameter tuning on model performance.

Each subdirectory provides a comprehensive set of visualizations that illustrate the clustering results and the evaluation metrics, enabling a detailed comparison of the different models and projections.

## Python Files
Following the initial exploration, the code was modularized into organized Python files for improved usability and maintainability.

### `main.py`
Handles the loading of the dataset from a CSV file, defines clustering parameters (`clusters` and `min_samples`), and performs KMeans and DBSCAN clustering. Outputs evaluation scores and generates relevant plots.

### `driver.py`
Initializes score arrays and orchestrates clustering based on the specified projection. Returns the computed score arrays.

### `clustering.py`
Implements clustering logic for both KMeans and DBSCAN. Supports clustering on PCA and t-SNE projections, resulting in four model options.

### `projection.py`
Contains functions for generating PCA and t-SNE projections. Returns the transformed datasets.

### `scores.py`
Provides functions for calculating evaluation metrics, including inertia and silhouette scores. Returns the computed scores in a structured format.

### `visualization.py`
Includes various functions for generating plots, including those for projections, clustering results, and evaluation metrics.

---

This project demonstrates the application of clustering techniques and dimensionality reduction methods to a real-world dataset, providing insights into the data and the effectiveness of different clustering approaches.