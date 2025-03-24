import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from main.projection import PCA_projection, TSNE_projection

def KMeans_clustering(data, clusters, projection):
    '''
    data: pandas DataFrame
    clusters: list of integers
    projection: string corresponding to the projection method
    return: KMeans clustering object
    1. If projection is PCA, project the data using PCA_projection
    2. If projection is tSNE, project the data using TSNE_projection
    3. For each number of clusters in the clusters list, fit a KMeans clustering object
    4. Calculate the inertia of the clustering
    5. Generate an inertia plot
    6. Print the inertia for each number of clusters
    7. Return the clustering object
    '''
    if projection == 'PCA':
        PCA_clusters = []
        for c in clusters:
            PCA_clusters.append(c)
            clustering = PCA_projection(data, c)
            return clustering
    elif projection == 'tSNE':
        tSNE_clusters = []
        for c in clusters:
            tSNE_clusters.append(c)
            clustering = TSNE_projection(data, c)
            return clustering
    else:
        print('Invalid projection method')
        return None


def DBSCAN_clustering(data, eps, min_samples, projection):
    '''
    data: pandas DataFrame
    eps: float
    min_samples: int
    return: DBSCAN clustering object
    1. If projection is PCA, project the data using PCA_projection
    2. If projection is tSNE, project the data using TSNE_projection
    3. Fit a DBSCAN clustering object
    4. Return the clustering object
    '''
    if projection == "PCA":
        PCA_min_samples = []
        for m in min_samples:
            PCA_min_samples.append(m)
            clustering = PCA_projection(data, m)
            return clustering
    elif projection == "tSNE":
        tSNE_min_samples = []
        for m in min_samples:
            tSNE_min_samples.append(m)
            clustering = TSNE_projection(data, m)
            return clustering
    else:
        print('Invalid projection method')
        return None 
    