import matplotlib.pyplot as plt
import numpy as np
import sys

class LinearRegression():
    def __init__(self, learning_rate, no_samples, no_epochs, weights):
        self.lr = learning_rate
        self.no_epochs = no_epochs
        self.weights = weights
        self.no_samples = no_samples
        
    def fit(self, X_train, Y_train):
        loss_his = []
        for epoch in range(self.no_epochs):
            Y_pred = self.predict(X_train)
            loss = self.cal_loss(Y_pred, Y_train)
            self.weights -= self.lr * np.dot(X_train.T, (Y_pred - Y_train))/self.no_samples
            loss_his.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.no_epochs}[", '='*20, ']')
                print("Loss: ", loss)
            
        return loss_his, self.weights
    
    def predict(self, X):
        return np.dot(X, self.weights)
    
    def cal_loss(self, Y_pred, label):
        loss = np.sum(np.square(Y_pred - label))
        return loss/(2*self.no_samples)
        
if __name__ == "__main__":
    np.random.seed(10)
    no_samples = 100
    mean = [2, 3]
    cov = [[2, 1.7], [1.7, 1.5]]
    data = np.random.multivariate_normal(mean, cov, no_samples)
    no_samples, no_features = data.shape
    
    X_train = np.concatenate((data[:, [0]], np.ones((no_samples, 1))), axis = 1)
    Y_train = data[:, [1]]
    
    #   Visualize data
    plt.scatter(data[:, 0], data[:, 1])
    plt.axis([-2, 6, -1, 6])
    
    #   Set hyperparameters
    learning_rate = 1e-3
    no_epochs = 1000
    
    model = LinearRegression(learning_rate, no_samples, no_epochs, weights)
    loss_his, weights = model.fit(X_train, Y_train)
    print(weights)
    
    #   Visualize loss and predict after training model
    # plt.plot(loss_his)
    plt.plot((-2, 6), ((weights[0]*(-2) + weights[1]), (weights[0]*(6) + weights[1])), c = 'r')
    
    