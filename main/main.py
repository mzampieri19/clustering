import pandas as pd
import numpy as np
from main.driver import perform_dbscan_clustering, perform_kmeans_clustering
from main.visualization import generate_silhouette_plot, generate_inirtia_plot
from scores import calculate_scores

def load_data(filepath):
    '''
    filepath: string
    return: pandas DataFrame
    1. Load the data from the filepath
    2. Return the data
    '''
    return pd.read_csv(filepath)

def main():
    '''
    1. Load the dataset
    2. Define parameters
    3. Perform KMeans clustering
    4. Perform DBSCAN clustering
    5. Calculate and print evaluation scores
    6. Generate plots
    '''
    # Load the dataset
    data = load_data('data/Income Survey Dataset.csv')

    # Define parameters
    clusters = np.arange(2, 11)
    min_samples = np.arange(5, 11)

    # Perform KMeans clustering
    silhouette_scores_kmeans, inertia_kmeans = perform_kmeans_clustering(data, clusters)

    # Perform DBSCAN clustering
    silhouette_scores_dbscan = perform_dbscan_clustering(data, min_samples)

    # Calculate and print evaluation scores
    scores = calculate_scores(data)
    print(scores)

    # pretty print all metrics
    print("KMeans Silhouette Scores: ", silhouette_scores_kmeans)
    print("KMeans Inertia Scores: ", inertia_kmeans)
    print("DBSCAN Silhouette Scores: ", silhouette_scores_dbscan)
    print("Evaluation Scores: ", scores)

    # Generate plots
    generate_silhouette_plot(clusters, silhouette_scores_kmeans, 'Silhouette Scores for KMeans Clustering')
    generate_inirtia_plot(clusters, inertia_kmeans, 'Inertia for KMeans Clustering')
    generate_silhouette_plot(min_samples, silhouette_scores_dbscan, 'Silhouette Scores for DBSCAN Clustering')


if __name__ == "__main__":
    main()