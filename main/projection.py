
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from main.visualization import generate_PCA_plot, generate_TSNE_plot

def PCA_projection(data, n_components):
    '''
    data: pandas DataFrame
    n_components: integer
    return: numpy array
    1. Create a PCA object with n_components
    2. Fit the PCA object to the data
    3. Transform the data using the PCA object
    4. Generate a plot of the transformed data
    5. Return the transformed data
    '''
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    generate_PCA_plot(transformed_data, 'PCA Projection Plot')
    return transformed_data

def TSNE_projection(data, n_components):
    '''
    data: pandas DataFrame
    n_components: integer
    return: numpy array
    1. Create a t-SNE object with n_components
    2. Fit the t-SNE object to the data
    3. Transform the data using the t-SNE object
    4. Generate a plot of the transformed data
    5. Return the transformed data
    '''
    tsne = TSNE(n_components=n_components)
    transformed_data = tsne.fit_transform(data)
    generate_TSNE_plot(transformed_data, 't-SNE Projection Plot')
    return transformed_data

