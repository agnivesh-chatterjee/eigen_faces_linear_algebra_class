import numpy as np

class EigenfaceRecognizer:
    """
    Nearest-neighbor classifier in eigenface space
    """

    def __init__(self, eigenface_model):
        self.model = eigenface_model
        self.omega_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Project training images into eigenface space
        """
        self.y_train = np.array(y_train)
        self.omega_train = self.model.project(X_train)

    def predict(self, X_test):
        """
        Predict labels for test images
        """
        omega_test = self.model.project(X_test)
        predictions = []

        for i in range(omega_test.shape[1]):
            # Compute Euclidean distance to all training projections
            diff = self.omega_train - omega_test[:, i:i+1]
            distances = np.linalg.norm(diff, axis=0)

            # Nearest neighbor
            predictions.append(self.y_train[np.argmin(distances)])

        return np.array(predictions)
