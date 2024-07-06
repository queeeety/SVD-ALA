import numpy as np

### TASK 1 ###
def compute_svd(A):
    # Compute the eigenvalues and eigenvectors
    eigvals_AAT, eigvecs_AAT = np.linalg.eig(A @ A.T)
    eigvals_ATA, eigvecs_ATA = np.linalg.eig(A.T @ A)

    # Sort the eigenvalues and eigenvectors in descending order
    idx_AAT = np.argsort(eigvals_AAT)[::-1]
    eigvals_AAT = eigvals_AAT[idx_AAT]
    U = eigvecs_AAT[:, idx_AAT]

    idx_ATA = np.argsort(eigvals_ATA)[::-1]
    eigvals_ATA = eigvals_ATA[idx_ATA]
    V = eigvecs_ATA[:, idx_ATA]

    # Create the Sigma matrix from the square root of the eigenvalues of A @ A.T
    Sigma = np.zeros(A.shape)
    Sigma[:A.shape[0], :A.shape[0]] = np.diag(np.sqrt(eigvals_AAT))

    return U, Sigma, V

# Test the function
A = np.array([[4, 2, 0], [1, 5, 6]])
U, Sigma, V = compute_svd(A)

# Check the decomposition by reconstructing the original matrix
A_reconstructed = np.round(U @ Sigma @ V.T, 2)

# Print the original and reconstructed matrices
print("Original Matrix:\n", A)
print("Reconstructed Matrix:\n", A_reconstructed)