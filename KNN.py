import numpy as np

class KNearestNeighbor():
    def __init__(self, k):
        self.k = k
    
    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y
    
    def predict(self, X_test):
        distances = self.compute_distance(X_test)
        return self.predict_label(distances)
    
    def compute_distance(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            distances[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train), axis = 1))
        return distances
    
    def predict_label(self, distances):
        num_test = distances.shape[0]
        Y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            indices = np.argsort(distances[i, :])
            k_closest_classes = self.Y_train[indices[:self.k]].astype(np.uint8)
            Y_pred[i] = np.argmax(np.bincount(k_closest_classes))
        return Y_pred
                
if __name__ == '__main__':
    X = np.loadtxt('Datasets\\data.txt', delimiter = ',')
    Y = np.loadtxt('Datasets\\targets.txt')
    KNN =  KNearestNeighbor(k = 3)
    KNN.train(X, Y)
    Y_pred = KNN.predict(X)
    print(f"Accuracy: {np.sum(Y_pred == Y)/Y.shape[0]*100:.2f}")