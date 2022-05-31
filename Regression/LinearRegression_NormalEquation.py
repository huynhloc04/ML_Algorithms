#   https://www.kaggle.com/phamdinhkhanh/singular-value-decomposition
import numpy as np
import matplotlib.pyplot as plt

def LinearRegression_NormalEquation(X, label):
    A = np.dot(X.T, X)
    b = np.dot(X.T, label)
    
    U, S, V_t = np.linalg.svd(A)
    S = np.diag(S)
    S_inv = np.linalg.inv(S)
    
    #   Calculate A_pseu
    A_pseu = np.dot(V_t.T, np.dot(S_inv, U.T))
    
    weights = np.dot(A_pseu, b)
    return weights

if __name__ == "__main__":
    np.random.seed(10)
    no_samples = 100
    mean = [2, 3]
    cov = [[2, 1.7], [1.7, 1.5]]
    data = np.random.multivariate_normal(mean, cov, no_samples)
    no_samples, no_features = data.shape
    
    X_train = np.concatenate((data[:, [0]], np.ones((no_samples, 1))), axis = 1)
    Y_train = data[:, [1]]
    plt.scatter(data[:, 0], data[:, 1])
    plt.axis([-2, 6, -1, 6])
    
    weights = LinearRegression_NormalEquation(X_train, Y_train)
    print(weights)
    plt.plot((-2, 6), ((weights[0]*(-2) + weights[1]), (weights[0]*(6) + weights[1])), c = 'r', linewidth = 2.5)
    