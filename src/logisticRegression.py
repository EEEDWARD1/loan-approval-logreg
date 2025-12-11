import numpy as np

class logisticRegression:
    def __init__(self):
        pass

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.
        :Param z: float or np.ndarray
            The weighted sum of input features (θ^T X).
        :Returns: float or np.ndarray
            The sigmoid value(s) of the input.
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def calculate_gradient(self, theta, X, y):
        """
        Compute the gradient of the cost function for logistic regression.
        :Param theta: np.ndarray
            The parameters of the logistic regression model.
        :Param X: np.ndarray
            The input feature matrix (with bias term).
        :Param y: np.ndarray
            The true labels.
        :Returns: np.ndarray
            The gradient of the cost function.
        """
        m = y.shape[0] # number of instances
        return (X.T @ (self.sigmoid(X @ theta) - y)) / m
    
    def gradient_descent(self, X, y, alpha = 0.01, num_iter = 100, tol=1e-7):
        """
        Perform gradient descent to learn the parameters of the logistic regression model.

        :Param X: np.ndarray
            The input feature matrix.
        :Param y: np.ndarray
            The true labels.
        :Param alpha: float
            The learning rate.
        :Param num_iter: int
            The maximum number of iterations.
        :Param tol: float
            The tolerance for convergence.
        :Returns: np.ndarray
            The learned parameters of the logistic regression model.
        
        Notes:
        - Adds a bias term to the input features.
        - Stops early if the gradient norm is below the specified tolerance.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # add bias term
        theta = np.zeros(X_b.shape[1])

        for i in range(num_iter):
            grad = self.calculate_gradient(theta, X_b, y)
            theta -= alpha * grad

            if np.linalg.norm(grad) < tol:
                break

        return theta
    
    def predict_proba(self, X, theta):
        """
        Predict the probabilities of the positive class for input features X.
        :Param X: np.ndarray
            The input feature matrix.
        :Param theta: np.ndarray
            The parameters of the logistic regression model.
        :Returns: np.ndarray
            The predicted probabilities of the positive class.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # add bias term
        return self.sigmoid(X_b @ theta )
    
    def predict(self, X, theta, threshold=0.5):
        """
        Predict binary class labels for input features X based on a threshold.
        :Param X: np.ndarray
            The input feature matrix.   
        :Param theta: np.ndarray
            The parameters of the logistic regression model.
        :Param threshold: float
            The threshold for classifying probabilities into binary labels.
        :Returns: np.ndarray
            The predicted binary class labels (0 or 1).
        """
        return (self.predict_proba(X, theta) >= threshold).astype(int)