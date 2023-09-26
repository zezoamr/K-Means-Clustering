import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClustering():
    
    def __init__(self, X, number_of_clusters, max_iterations):
        self.k = number_of_clusters
        self.max_iterations = max_iterations
        self.plot_figure = True
        self.number_of_examples, self.number_of_features = X.shape
        
    def initialize_centroids(self, X):
        return X[np.random.choice(X.shape[0], self.k, replace=False)]
    
    def create_clusters(self, X, centroids):
        clusters = [[] for i in range(self.k)] #contains indicies of points in clusters
        for idx, point in enumerate(X):
            closest_centroid = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])
            clusters[closest_centroid].append(idx)
        return clusters
    
    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.k, self.number_of_features))
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(X[cluster], axis=0)
        return centroids
    
    def perdict_cluster(self, clusters):
        #returns each point with the cluster it belongs to
        y_pred = np.zeros(self.number_of_examples)
        
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                y_pred[point] = idx
        return y_pred
    
    def plot_fig(self, X, y):
        # X is the data, y is the labels of which cluster the point belongs to
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    
    def fit(self, X):
        centroids = self.initialize_centroids(X)
        for i in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            prev_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            if np.array_equal(centroids, prev_centroids):
                print("Converged")
                break
            
        y_pred = self.perdict_cluster(clusters)
        if self.plot_figure:
            self.plot_fig(X, y_pred)
        return y_pred
    
    
def main():
    np.random.seed(10)
    blob_num_clusters = 5
    k = 2
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=blob_num_clusters)

    Kmeans = KMeansClustering(X, k, 100)
    y_pred = Kmeans.fit(X)

if __name__ == "__main__":
    main()