
import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering():
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.no_samples = X.shape[0]
    
    def centroids_init(self):
        #   Choice 3 data points to as 3 centroids initialization
        centroids = self.X[np.random.choice(self.no_samples, self.K, replace = False)]
        return centroids
    
    def cal_distances(self, centroids):
        distances = np.zeros((self.no_samples, self.K))
        for i in range(self.no_samples):
            distances[i] = np.linalg.norm((self.X[i] - centroids), axis = 1)
        return distances
    
    def assign_labels(self, centroids):
        distances = self.cal_distances(centroids)
        return np.argmin(distances, axis = 1)
    
    def update_centroids(self, labels):
        centroids = np.zeros((self.K, self.X.shape[1]))
        for k in range(self.K):
            centroids[k] = np.mean(self.X[labels == k], axis = 0)
        return centroids
            
    def fit(self):
        centroids_lst = [self.centroids_init()]
        labels = []
        while True:
            #   Assign labels for all data points with each it's mean
            labels.append(self.assign_labels(centroids_lst[-1]))
            #   Calculate and update new centroids 
            new_centroids = self.update_centroids(labels[-1])
            #   Check if coordinate of new_centroids and old_centroids are equal => Stop updating new centroids
            if np.array_equal(new_centroids, centroids_lst[-1]):
                break
            centroids_lst.append(new_centroids)
        return centroids_lst[-1], labels[-1]
    
    def visualize(self, labels, state):        
        if state == 'result':
            X1 = self.X[labels == 0]
            X2 = self.X[labels == 1]
            X3 = self.X[labels == 2]
            
            plt.scatter(X1[:, 0], X1[:, 1], c = 'r', marker = 's', alpha = 0.7)
            plt.scatter(X2[:, 0], X2[:, 1], c = 'g', marker = 'd', alpha = 0.7)
            plt.scatter(X3[:, 0], X3[:, 1], c = 'y', marker = 'o', alpha = 0.7)
        else:
            plt.scatter(self.X[:, 0], self.X[:, 1])
        plt.show()
        
                    
if __name__ == "__main__":
    
    np.random.seed(4)
    n_samples = 500
    means = [[3, 3], [-3, 4], [0, -3]]
    cov = [[1.5, 0], [0, 2]]
    X1 = np.random.multivariate_normal(means[0], cov, n_samples)
    X2 = np.random.multivariate_normal(means[1], cov, n_samples)
    X3 = np.random.multivariate_normal(means[2], cov, n_samples)
    data = np.concatenate((X1, X2, X3), axis = 0)
    num_class = 3
    
    model = KMeansClustering(data, num_class)
    centers, labels = model.fit()
    print('Trained centroids\n', centers)
    
    model.visualize(labels, 'result')
    