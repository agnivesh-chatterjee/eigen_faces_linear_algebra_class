import numpy as np

class Eigenfaces:
    """
    Eigenfaces implementation (Turk & Pentland, 1991)

    X is expected to be of shape:
        (num_pixels, num_images)
    """

    def __init__(self, num_components):
        self.K = num_components
        self.mean_face = None
        self.eigenfaces = None

    def fit(self, X):
        """
        Compute eigenfaces from training data
        """

        # ---------- SAFETY CHECK ----------
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        # Mean face (Ψ)
        self.mean_face = np.mean(X, axis=1, keepdims=True)

        # Mean-centered data matrix A
        A = X - self.mean_face

        # Covariance trick: L = AᵀA  (M x M)
        L = A.T @ A

        # Eigen-decomposition of L
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Compute eigenfaces u_k = A v_k
        eigenfaces = A @ eigenvectors

        # Keep only top K
        eigenfaces = eigenfaces[:, :self.K]

        # ---------- CRITICAL FIX ----------
        # Ensure eigenfaces is ALWAYS 2D
        if eigenfaces.ndim == 1:
            eigenfaces = eigenfaces.reshape(-1, 1)

        # Normalize eigenfaces (column-wise)
        norms = np.linalg.norm(eigenfaces, axis=0, keepdims=True)
        self.eigenfaces = eigenfaces / norms

    def project(self, X):
        """
        Project images into eigenface space
        """

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        A = X - self.mean_face
        return self.eigenfaces.T @ A

    def reconstruct(self, omega):
        """
        Reconstruct image from eigenface coefficients
        """

        if omega.ndim == 1:
            omega = omega.reshape(-1, 1)

        return self.mean_face + self.eigenfaces @ omega
