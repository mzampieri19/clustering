from main.clustering import DBSCAN_clustering, KMeans_clustering
from main.scores import silhouette_score
from main.visualization import generate_PCA_plot, generate_TSNE_plot


def perform_kmeans_clustering(data, clusters):
    '''
    data: pandas DataFrame
    clusters: list of integers
    return: list of silhouette scores
    1. For each number of clusters in the clusters list, perform KMeans clustering
    2. Generate a PCA plot of the clustering
    3. Generate a t-SNE plot of the clustering
    4. Calculate the silhouette score of the clustering
    5. Append the silhouette score to a list
    6. Return the list of silhouette scores
    '''
    silhouette_scores = []
    inertia_scores = []

    for cluster in clusters:
        # PCA projection
        pca_kmeans = KMeans_clustering(data, cluster, projection='PCA')
        generate_PCA_plot(pca_kmeans, 'PCA KMeans Clustering')

        # t-SNE projection
        tsne_kmeans = KMeans_clustering(data, cluster, projection='tSNE')
        generate_TSNE_plot(tsne_kmeans, 'tSNE KMeans Clustering')

        # Calculate silhouette score and inertia
        kmeans_model = KMeans_clustering(data, cluster)  # Assuming this returns the fitted model
        labels = kmeans_model.labels_
        silhouette_scores.append(silhouette_score(data, labels))
        inertia_scores.append(kmeans_model.inertia_)

    return silhouette_scores, inertia_scores

def perform_dbscan_clustering(data, min_samples, eps=0.5):
    '''
    data: pandas DataFrame
    min_samples: list of integers
    eps: float
    return: list of silhouette scores
    1. For each number of min_samples in the min_samples list, perform DBSCAN clustering
    2. Generate a PCA plot of the clustering
    3. Generate a t-SNE plot of the clustering
    4. Calculate the silhouette score of the clustering
    5. Append the silhouette score to a list
    6. Return the list of silhouette scores
    '''
    silhouette_scores = []

    for sample in min_samples:
        # PCA projection
        pca_dbscan = DBSCAN_clustering(data, eps=eps, min_samples=sample, projection='PCA')
        generate_PCA_plot(pca_dbscan, 'PCA DBSCAN Clustering')

        # t-SNE projection
        tsne_dbscan = DBSCAN_clustering(data, eps=eps, min_samples=sample, projection='tSNE')
        generate_TSNE_plot(tsne_dbscan, 'tSNE DBSCAN Clustering')

        # Calculate silhouette score
        dbscan_model = DBSCAN_clustering(data, eps=eps, min_samples=sample)  # Assuming this returns the fitted model
        labels = dbscan_model.labels_
        if len(set(labels)) > 1:  # Silhouette score requires at least 2 clusters
            silhouette_scores.append(silhouette_score(data, labels))
        else:
            silhouette_scores.append(None)  # Append None if silhouette score cannot be calculated

    return silhouette_scores