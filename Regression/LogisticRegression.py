import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticRegression():
    def __init__(self, learning_rate, no_epochs, no_samples, weights):
        self.lr = learning_rate
        self.no_epochs = no_epochs
        self.no_samples = no_samples
        self.weights = weights
    
    def fit(self, X_train, Y_train):
        loss_his = []
        for epoch in range(1, self.no_epochs + 1):
            pred = self.predict(X_train)
            loss = self.loss_func(pred, Y_train)
            self.weights -= self.lr * np.dot(X_train.T, (pred - Y_train))/self.no_samples
            loss_his.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.no_epochs}[", '='*20, ']')
                print("Loss: ", loss)
            
        return loss_his, weights
    
    def predict(self, X_pred, state = None):
        pred =  self.sigmoid_func(np.dot(X_pred, self.weights))
        if state == 'test':
            return 1 if pred > 0.5 else 0
        return pred
    
    def sigmoid_func(self, z):
        return 1/(1 + np.exp(-z))
    
    def loss_func(self, Y_pred, label):
        loss = -np.sum(label*(np.log(Y_pred) + (1-label)*np.log(1-Y_pred)))
        return loss/self.no_samples
            
if __name__ == "__main__":
    np.random.seed(12)
    X, Y = make_blobs(n_samples = 1000, centers = 2)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    no_samples, no_features = X.shape
    Y = np.expand_dims(Y, axis = 1)
    weights = np.random.randn(no_features, 1)
    #   Set some hyperparameters
    learning_rate = 1e-3
    no_epochs = 2000
    
    model = LogisticRegression(learning_rate, no_epochs, no_samples, weights)
    loss_his, weight_updated = model.fit(X, Y)
    print(weights)
    plt.plot(loss_his)
    

        
    
    
    
    
    
    
    