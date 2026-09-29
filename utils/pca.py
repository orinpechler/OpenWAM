import numpy as np


def pca_2d(priors: np.ndarray) -> np.ndarray:
    """Flatten (N, 32, 80) action priors to (N, 2560) and return their (N, 2) PCA projection."""
    x = priors.reshape(priors.shape[0], -1).astype(np.float64)
    x -= x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T
