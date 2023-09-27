import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClustering():
    
    def __init__(self, X, number_of_clusters, max_iterations, plot_figure):
        self.k = number_of_clusters
        self.max_iterations = max_iterations
        self.plot_figure = plot_figure
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
    
class KMeans():
    def __init__(self, X, max_number_of_clusters, max_iterations):
        self.max_number_of_clusters = max_number_of_clusters
        self.kmeans_arr = [KMeansClustering(X, i, max_iterations, False) for i in range(1, max_number_of_clusters+1)]
        self.max_iterations = max_iterations
    
    def calculate_variance(self, X, clusters):
        variance = [0 for _ in range(self.max_number_of_clusters)]
        for idx, cluster in enumerate(clusters):
            variance[idx] = np.var(X[cluster], axis=0).sum()
        return variance
            
    def compare_variance(self, iteration_variance):
        # Convert the iteration_variance list into a numpy array
        iteration_variance = np.array(iteration_variance)

        # Get the total variances
        total_variances = np.array([np.sum(x[0]) for x in iteration_variance])

        # Calculate the difference in total variances between each pair of consecutive K's
        diff_variances = np.diff(total_variances)

        # Calculate the difference in total variances between each pair of consecutive differences
        diff_diff_variances = np.diff(diff_variances)

        # The optimal K is where the second difference is maximum (i.e., the elbow point)
        best_k = np.argmax(diff_diff_variances) + 1

        return best_k
    
    def plot_fig(self, X, y_pred):
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=40, cmap=plt.cm.Spectral)
        plt.show()
    
    def fit(self, X):
        variance_arr = []
        y_pred_arr = []
        for idx, kmeans in enumerate(self.kmeans_arr):
            y_pred = kmeans.fit(X)
            y_pred_arr.append(y_pred)
            clusters = kmeans.create_clusters(X, kmeans.initialize_centroids(X))
            total_variance = np.sum(self.calculate_variance(X, clusters))  
            print("k: ", idx + 1)
            print("total variance: ", total_variance)
            variance_arr.append([total_variance, idx])
            
        best_k = self.compare_variance(variance_arr)
        return y_pred_arr[best_k], best_k + 1  # best_k + 1 because we started from 1 cluster
    
def main():
    np.random.seed(10)
    blob_num_clusters = 5
    k = 8
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=blob_num_clusters)

    #Kmeans = KMeansClustering(X, k, 100, True)
    Kmeans = KMeans(X, k, 100)
    y_pred, k = Kmeans.fit(X)
    print("best k: ", k)
    Kmeans.plot_fig(X, y_pred)

if __name__ == "__main__":
    main()