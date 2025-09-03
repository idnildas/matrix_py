# Singular Value Decomposition (SVD) Theory

# SVD is a fundamental matrix factorization technique in linear algebra.
# For any real or complex matrix A of size m x n, SVD states that A can be decomposed as:
#     A = U * Σ * V^T
# where:
#   - U is an m x m orthogonal (or unitary) matrix. Its columns are called the left singular vectors of A.
#   - Σ (Sigma) is an m x n diagonal matrix with non-negative real numbers on the diagonal, called singular values.
#   - V is an n x n orthogonal (or unitary) matrix. Its columns are called the right singular vectors of A.
#   - V^T is the transpose (or conjugate transpose) of V.

# Properties:
# - The singular values in Σ are the square roots of the eigenvalues of A^T A (or AA^T).
# - The number of non-zero singular values equals the rank of A.
# - SVD exists for any real or complex matrix, regardless of its shape.
# - SVD is widely used for dimensionality reduction, noise reduction, and solving linear inverse problems.

# Applications:
# - Principal Component Analysis (PCA)
# - Signal processing
# - Image compression
# - Latent Semantic Analysis (LSA) in Natural Language Processing

# Example:
# For a matrix A:
#     A = [[a11, a12],
#          [a21, a22],
#          [a31, a32]]
# SVD finds matrices U, Σ, and V such that A = U * Σ * V^T.

import math
import matrix_transpose
import matrix_multiply
import matrix_qr_eigenvalues
import numpy as np

def gram_schmidt_columns(matrix):
    """
    Orthonormalizes the columns of a matrix using the Gram-Schmidt process.
    
    Args:
        matrix: Input matrix (list of lists) where each column should be orthonormalized
        
    Returns:
        Matrix with orthonormal columns
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0
    tol = 1e-12
    
    # Extract columns
    columns = []
    for j in range(n_cols):
        col = [matrix[i][j] for i in range(n_rows)]
        columns.append(col)
    
    # Gram-Schmidt orthogonalization
    orthonormal_cols = []
    for j in range(n_cols):
        # Start with current column
        v = columns[j][:]
        
        # Subtract projections onto previous orthonormal vectors
        for q in orthonormal_cols:
            # Compute projection coefficient: <v, q>
            proj_coeff = sum(v[i] * q[i] for i in range(n_rows))
            # Subtract projection: v = v - <v, q> * q
            for i in range(n_rows):
                v[i] -= proj_coeff * q[i]
        
        # Normalize the result
        norm = math.sqrt(sum(x**2 for x in v))
        if norm > tol:
            # Normalize to unit vector
            v = [x / norm for x in v]
        else:
            # Handle zero/nearly zero vector - use standard basis
            v = [0.0] * n_rows
            if j < n_rows:
                v[j] = 1.0
            
            # Re-orthogonalize against existing vectors
            for q in orthonormal_cols:
                proj_coeff = sum(v[i] * q[i] for i in range(n_rows))
                for i in range(n_rows):
                    v[i] -= proj_coeff * q[i]
            
            # Renormalize
            norm = math.sqrt(sum(x**2 for x in v))
            if norm > tol:
                v = [x / norm for x in v]
        
        orthonormal_cols.append(v)
    
    # Convert back to matrix format (rows x columns)
    result = [[orthonormal_cols[j][i] for j in range(n_cols)] for i in range(n_rows)]
    return result

def svd(matrix, compute_uv=True, full_matrices=True, precision=6, verbose=False):
    """
    Industry-ready Singular Value Decomposition (SVD) with comprehensive error handling.
    
    For matrix A (m x n), computes A = U @ Σ @ V^T where:
    - U: m x m orthogonal matrix (left singular vectors)
    - Σ: m x n diagonal matrix (singular values)
    - V^T: n x n orthogonal matrix (right singular vectors transposed)
    
    Args:
        matrix: Input matrix A (list of lists)
        compute_uv: If True, compute U and V matrices; if False, only singular values
        full_matrices: If True, return full U and V; if False, return economy-size
        precision: Number of decimal places for rounding singular values
        verbose: If True, print detailed comparison with NumPy
        
    Returns:
        tuple: (U, Sigma, V_transpose) if compute_uv=True, else just singular_values
        
    Raises:
        ValueError: For invalid inputs
        RuntimeError: For numerical failures
    """
    # Input validation
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")
    
    m = len(matrix)
    n = len(matrix[0])
    
    if any(len(row) != n for row in matrix):
        raise ValueError("All matrix rows must have the same length")
    
    if any(not isinstance(x, (int, float)) for row in matrix for x in row):
        raise ValueError("Matrix elements must be numeric")
    
    # Check for numerical issues
    max_element = max(abs(x) for row in matrix for x in row)
    if max_element > 1e15:
        raise ValueError("Matrix elements too large - risk of overflow")
    
    if max_element < 1e-15:
        return _handle_zero_matrix(m, n, compute_uv, full_matrices)
    
    # Perform SVD computation
    try:
        A = [row[:] for row in matrix]
        tol = max(1e-12, max_element * 1e-15)  # Adaptive tolerance
        
        # Step 1: Compute A^T A
        At = matrix_transpose.transpose_matrix(A)
        AtA = matrix_multiply.matrix_multiply(At, A)
        
        # Step 2: Robust eigenvalue computation
        eigenvalues_AtA, eigenvectors_AtA = matrix_qr_eigenvalues.qr_eigenvalues_with_vectors(AtA)
        
        # Step 3: Extract and validate singular values
        sv_data = []
        for i, (ev, eigvec) in enumerate(zip(eigenvalues_AtA, eigenvectors_AtA)):
            if ev > tol:
                sv = math.sqrt(ev)
                if math.isnan(sv) or math.isinf(sv):
                    raise RuntimeError(f"Invalid singular value computed: {sv}")
                sv_data.append((sv, eigvec, i))
        
        if not sv_data:
            return _handle_zero_matrix(m, n, compute_uv, full_matrices)
        
        # Sort by singular value (descending)
        sv_data.sort(key=lambda x: x[0], reverse=True)
        singular_values = [round(sv, precision) for sv, _, _ in sv_data]
        
        if not compute_uv:
            return singular_values
        
        r = len(singular_values)  # Numerical rank
        
        # Determine matrix sizes based on full_matrices flag
        if full_matrices:
            u_cols, v_rows = m, n
        else:
            u_cols, v_rows = min(m, r), min(n, r)
        
        # Build V matrix
        V = _build_v_matrix(sv_data, eigenvectors_AtA, n, v_rows, r, tol)
        
        # Build U matrix
        U = _build_u_matrix(A, V, singular_values, m, u_cols, r, tol)
        
        # Build Sigma matrix
        if full_matrices:
            Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
            for i in range(r):
                Sigma[i][i] = singular_values[i]
        else:
            min_dim = min(m, n)
            Sigma = [[0.0 for _ in range(min_dim)] for _ in range(min_dim)]
            for i in range(min(r, min_dim)):
                Sigma[i][i] = singular_values[i]
        
        V_transpose = matrix_transpose.transpose_matrix(V)
        
        # Verification
        _verify_svd(A, U, Sigma, V_transpose, tol * 1e3)
        
        # Optional verbose output for debugging/comparison
        if verbose:
            _verbose_output(A, U, Sigma, V_transpose, singular_values, m, n)
        
        return U, Sigma, V_transpose
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"SVD computation failed: {str(e)}")

def _verbose_output(A, U, Sigma, V_transpose, singular_values, m, n):
    """Generate verbose output comparing with NumPy."""
    try:
        # Reconstruct A = U @ Σ @ V^T
        US = matrix_multiply.matrix_multiply(U, Sigma)
        A_reconstructed = matrix_multiply.matrix_multiply(US, V_transpose)
        
        print("=== Custom SVD Results ===")
        print("Singular values:", [round(sv, 6) for sv in singular_values])
        print("Matrix dimensions: A{}  U{}  Σ{}  V^T{}".format(
            (m, n), (len(U), len(U[0])), (len(Sigma), len(Sigma[0])), 
            (len(V_transpose), len(V_transpose[0]))))
        
        # Compute reconstruction error
        max_error = 0.0
        for i in range(m):
            for j in range(n):
                error = abs(A[i][j] - A_reconstructed[i][j])
                max_error = max(max_error, error)
        
        print("Reconstruction max error:", max_error)
        
        # Compare with NumPy
        np_A = np.array(A)
        np_U, np_s, np_Vt = np.linalg.svd(np_A, full_matrices=True)
        
        print("\n=== NumPy Comparison ===")
        print("NumPy singular values:", [round(sv, 6) for sv in np_s])
        
        # Check singular value agreement (allow small differences)
        sv_agreement = all(abs(sv1 - sv2) < 1e-3 
                          for sv1, sv2 in zip(singular_values, np_s[:len(singular_values)]))
        print("Singular values agree with NumPy:", sv_agreement)
        
        # NumPy reconstruction  
        np_Sigma_full = np.zeros((m, n))
        for i in range(len(np_s)):
            np_Sigma_full[i, i] = np_s[i]
        np_reconstructed = np.dot(np.dot(np_U, np_Sigma_full), np_Vt)
        
        numpy_max_error = np.max(np.abs(np_A - np_reconstructed))
        print("NumPy reconstruction max error:", numpy_max_error)
        
        if max_error > 1e-6:
            print("WARNING: High reconstruction error detected!")
            print("This may indicate numerical instability in the algorithm.")
    except Exception as e:
        print(f"Warning: Could not generate verbose output: {e}")

def _handle_zero_matrix(m, n, compute_uv, full_matrices):
    """Handle the special case of zero/near-zero matrices."""
    if not compute_uv:
        return []
    
    # Return identity matrices for zero input
    U = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
    
    return U, Sigma, matrix_transpose.transpose_matrix(V)

def _build_v_matrix(sv_data, eigenvectors, n, v_rows, r, tol):
    """Build the V matrix from eigenvectors."""
    V = [[0.0 for _ in range(v_rows)] for _ in range(n)]
    
    # Place right singular vectors
    for j, (_, eigvec, _) in enumerate(sv_data[:v_rows]):
        norm = math.sqrt(sum(x**2 for x in eigvec))
        if norm > tol:
            for i in range(n):
                V[i][j] = eigvec[i] / norm
        else:
            for i in range(n):
                V[i][j] = 1.0 if i == j else 0.0
    
    # Fill remaining columns with orthogonal vectors
    V = gram_schmidt_columns(V)
    return V

def _build_u_matrix(A, V, singular_values, m, u_cols, r, tol):
    """Build the U matrix from A, V, and singular values."""
    U = [[0.0 for _ in range(u_cols)] for _ in range(m)]
    
    # Compute left singular vectors
    for j in range(min(r, u_cols)):
        v_col = [V[i][j] for i in range(len(V))]
        u_col = [sum(A[i][k] * v_col[k] for k in range(len(v_col))) for i in range(m)]
        
        # Normalize by singular value
        for i in range(m):
            U[i][j] = u_col[i] / singular_values[j]
    
    # Orthonormalize
    U = gram_schmidt_columns(U)
    return U

def _verify_svd(A, U, Sigma, V_transpose, tol):
    """Verify SVD reconstruction accuracy."""
    try:
        US = matrix_multiply.matrix_multiply(U, Sigma)
        A_reconstructed = matrix_multiply.matrix_multiply(US, V_transpose)
        
        max_error = 0.0
        for i in range(len(A)):
            for j in range(len(A[0])):
                error = abs(A[i][j] - A_reconstructed[i][j])
                max_error = max(max_error, error)
        
        # Use more reasonable tolerance for warnings
        reasonable_tol = max(tol, 1e-10)
        if max_error > reasonable_tol:
            print(f"Warning: SVD reconstruction error {max_error:.2e} exceeds tolerance {reasonable_tol:.2e}")
            
    except Exception:
        print("Warning: Could not verify SVD reconstruction")

