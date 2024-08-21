import numpy as np 
import pandas as pd 

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

learning_rate = 0.1
nb_epochs_default = 100

class LogisticRegression:
    
    def __init__(self, dim: int = 1):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        assert dim >= 1, f"dim should be >= 1 but now it {dim}, which is lowest than 1"
        self.dim = dim

    def init_weights(self, X, y, mode="auto"):
        match mode:
            case 'auto':
                self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
                self.b = 1
            case 'manual':
                W0, W1, self.b = 10, -5.08618309, -2.93316039
                self.weights = np.array([[W0, W1]]).reshape(2, 1)

        
    def fit(self, X, y, init_mode = "auto", lr = learning_rate, nb_epochs = nb_epochs_default, plot: bool = True):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """

        lr = lr / len(y)
        
        X = X[:, :, np.newaxis] ** np.arange(1, self.dim + 1)
        X = X.reshape(X.shape[0], -1)

        self.init_weights(X, y, mode=init_mode)
        
        accs, losss, hinge_losss = [], [], []

        for _ in range(nb_epochs):

            y_pred = sigmoid(X.dot(self.weights) + self.b)
            
            dw = np.dot(X.T, y - y_pred).reshape(-1, 1) 
            db = np.sum((y - y_pred))

            self.weights = self.weights + lr * dw
            self.b = self.b + lr * db

            accs.append(binary_accuracy(y, y_pred))
            losss.append(binary_cross_entropy(y, y_pred))
            hinge_losss.append(hinge_loss(y, y_pred))
        hinge_losss = np.array(hinge_losss).reshape(-1)

        if plot:
            plot_loss_and_acc(accs, losss)
    

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X = X[:, :, np.newaxis] ** np.arange(1, self.dim + 1)
        X = X.reshape(X.shape[0], -1)

        return sigmoid(X.dot(self.weights) + self.b)
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape, f"y_true.shape: {y_true.shape} and y_pred.shape: {y_pred.shape}"
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape, f"y_true.shape: {y_true.shape} and y_pred.shape: {y_pred.shape}"
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1 / (1 + np.exp(-x))


def hinge_loss(y_true, y_pred):
    y_true = np.where(y_true == 0, -1, 1)
    y_pred = np.where(y_pred == 0, -1, 1)
    return max(0, 1 - y_true.T.dot(y_pred) / ( y_true.shape[0] ** 2 ))


import matplotlib.pyplot as plt
def plot_loss_and_acc(accs, losss):
    nb_column = 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(losss, label='trainset loss')
    plt.legend()
    plt.subplot(1, nb_column, nb_column)
    plt.plot(accs, label='trainset accuracy')
    plt.legend()
    plt.show()