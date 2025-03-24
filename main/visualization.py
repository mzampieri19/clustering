import matplotlib.pyplot as plt

def generate_PCA_plot(data, title):
    '''
    data: numpy array
    title: string
    1. Create a scatter plot of the data
    2. Set the title of the plot
    3. Set the x-axis label to 'Principal Component 1'
    4. Set the y-axis label to 'Principal Component 2'
    5. Display the plot
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()

def generate_TSNE_plot(data, title):
    '''
    data: numpy array
    title: string
    1. Create a scatter plot of the data
    2. Set the title of the plot
    3. Set the x-axis label to 't-SNE Component 1'
    4. Set the y-axis label to 't-SNE Component 2'
    5. Display the plot
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    plt.show()

def generate_inirtia_plot(clusters, inirtia, title):
    '''
    data: numpy array
    title: string
    1. Create a line plot of the inertia values
    2. Set the title of the plot
    3. Set the x-axis label to 'Number of Clusters'
    4. Set the y-axis label to 'Inertia'
    5. Display the plot
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(clusters, inirtia, marker='o')
    plt.title(title)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()

def generate_silhouette_plot(clusters, silhouette_scores, title):
    '''
    data: numpy array
    title: string
    1. Create a line plot of the silhouette scores
    2. Set the title of the plot
    3. Set the x-axis label to 'Number of Clusters'
    4. Set the y-axis label to 'Silhouette Score'
    5. Display the plot
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(clusters, silhouette_scores, marker='o')
    plt.title(title)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()