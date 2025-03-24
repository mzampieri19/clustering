import numpy as np
from sklearn.metrics import silhouette_score as sk_silhouette_score

def calculate_inertia(clustering, data):
    '''
    clustering: KMeans/ DBSCAN clustering object
    data: numpy array
    return: float
    1. Get the cluster centers from the clustering object
    2. Calculate the inertia of the clustering
    3. Return the inertia
    '''
    cluster_centers = clustering.cluster_centers_
    inertia = np.sum((data - cluster_centers) ** 2)
    return inertia

def silhouette_score(clustering):
    '''
    clustering: KMeans/ DBSCAN clustering object
    return: float
    1. Get the labels from the clustering object
    2. Calculate the silhouette score of the clustering
    3. Return the silhouette score
    '''
    labels = clustering.labels_
    score = sk_silhouette_score(clustering, labels)
    return score

def calculate_scores(clustering):
    '''
    clustering: KMeans/ DBSCAN clustering object
    return: dictionary
    1. Get the cluster centers and labels from the clustering object
    2. Calculate the inertia of the clustering
    3. Calculate the silhouette score of the clustering
    4. Return a dictionary with the inertia and silhouette score
    '''
    cluster_centers = clustering.cluster_centers_
    labels = clustering.labels_
    inertia = calculate_inertia(cluster_centers, clustering)
    silhouette = silhouette_score(clustering, labels)
    return {
        'inertia': inertia,
        'silhouette_score': silhouette
    }