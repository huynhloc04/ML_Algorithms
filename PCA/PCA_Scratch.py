
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA():
    def __init__(self, n_comp):
        self.comp = None
        self.n_comp = n_comp

    def fit(self, X):
        """
            Args:
                X   Shape: [n_features, n_samples]
        """
        #   Calculate mean
        self.mean = np.mean(X, axis=1, keepdims=True)
        X = X - self.mean
        #   Calculate covariance matrix
        cov_mtr = np.cov(X)
        #   Calculate eigen_vectors & eigen_values
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mtr)
        #   Sort eigen_vals descending order
        idx_sort = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx_sort]
        eigen_vecs = eigen_vecs[:, idx_sort]         #   Shape: [n_features, n_features] (Each column is an eigen vector)
        #   Store first n_comp eigen_vectors
        self.n_comp = eigen_vecs[:, :self.n_comp]      #   Shape: [n_features, n_comp]
        

    def transform(self, X):
        X = X - self.mean   #   ReCalculate mean of Data
        return np.dot(self.n_comp.T, X)       #   Shape: [n_comp, n_samples]


data = datasets.load_iris()
X_data = data.data.T
Y_data = data.target

pca = PCA(n_comp=2)
vals = pca.fit(X_data)
X_proj = pca.transform(X_data)

plt.figure(figsize=(10, 8))
plt.scatter(X_proj[0], X_proj[1], c=Y_data, alpha=0.7)
plt.xlabel('PC1', fontsize=15, c='g')
plt.xlabel('PC2', fontsize=15, c='g')
plt.show()
