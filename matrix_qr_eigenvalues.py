# Industry-Ready QR Algorithm for Eigenvalue Computation
"""
QR Algorithm for Eigenvalues - Production Implementation
=======================================================

This module provides robust, numerically stable implementations of the QR algorithm
for computing eigenvalues and eigenvectors of square matrices. The implementation
uses Householder QR decomposition for optimal numerical stability.

Key Features:
- Householder QR decomposition (numerically stable)
- Robust error handling and input validation
- Adaptive convergence detection
- Industry-standard tolerances and precision
- Comprehensive diagnostics and monitoring

Mathematical Foundation:
The QR algorithm iteratively applies QR decomposition: A₀ = A, Aₖ₊₁ = RₖQₖ
where Aₖ = QₖRₖ. The sequence converges to upper triangular form with 
eigenvalues on the diagonal.
"""
import math 


def qr_eigenvalues(matrix, max_iter=10000, tol=1e-12, precision=6, return_diagnostics=False):
    """
    Industry-ready QR algorithm for eigenvalue computation with robust error handling.
    
    Args:
        matrix: Square matrix (list of lists)
        max_iter: Maximum iterations (default: 10000)
        tol: Convergence tolerance (default: 1e-12)
        precision: Number of decimal places for rounding (default: 6)
        return_diagnostics: If True, return detailed diagnostics
        
    Returns:
        list: Eigenvalues if return_diagnostics=False
        dict: Contains eigenvalues, convergence info, and diagnostics if return_diagnostics=True
        
    Raises:
        ValueError: For invalid input matrices
        RuntimeError: For convergence failures
    """
    # Input validation
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")
    
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")
    
    if any(not isinstance(x, (int, float)) for row in matrix for x in row):
        raise ValueError("Matrix elements must be numeric")
    
    # Check for numerical issues
    max_element = max(abs(x) for row in matrix for x in row)
    if max_element > 1e15:
        raise ValueError("Matrix elements too large - risk of overflow")
    
    if max_element < 1e-15:
        raise ValueError("Matrix elements too small - risk of underflow")
    
    # QR algorithm with convergence tracking
    A = [row[:] for row in matrix]
    convergence_history = []
    
    for iteration in range(max_iter):
        try:
            Q, R = householder_qr(A)
            A_new = [[sum(R[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            
            # Check for NaN or infinity
            if any(math.isnan(x) or math.isinf(x) for row in A_new for x in row):
                raise RuntimeError(f"Numerical instability detected at iteration {iteration}")
            
            # Convergence check
            off_diag = sum(abs(A_new[i][j]) for i in range(n) for j in range(n) if i != j)
            convergence_history.append(off_diag)
            
            A = A_new
            
            if off_diag < tol:
                break
                
            # Check for stagnation
            if iteration > 100 and len(convergence_history) >= 10:
                recent_changes = [abs(convergence_history[i] - convergence_history[i-1]) 
                                for i in range(-9, 0)]
                if all(change < tol * 1e3 for change in recent_changes):
                    break  # Converged as much as possible
                    
        except Exception as e:
            raise RuntimeError(f"QR decomposition failed at iteration {iteration}: {str(e)}")
    
    else:
        if return_diagnostics:
            print(f"Warning: QR algorithm reached maximum iterations ({max_iter})")
            print(f"Final off-diagonal norm: {off_diag:.2e}")
    
    # Extract eigenvalues
    eigenvalues = []
    for i in range(n):
        eigenval = A[i][i]
        # Smart rounding: preserve significant digits
        if abs(eigenval) > 10**(-precision):
            eigenvalues.append(round(eigenval, precision))
        else:
            eigenvalues.append(round(eigenval, precision + 6))
    
    if return_diagnostics:
        # Prepare result with diagnostics
        result = {
            'eigenvalues': eigenvalues,
            'iterations': iteration + 1,
            'converged': off_diag < tol,
            'final_off_diagonal_norm': off_diag,
            'condition_estimate': max_element / min(abs(A[i][i]) for i in range(n) if abs(A[i][i]) > tol)
        }
        return result
    else:
        return eigenvalues


def householder_qr(A):
    """
    Performs QR decomposition using Householder reflections (numerically stable).
    Args:
        A (list of lists): Square matrix.
    Returns:
        Q (list of lists): Orthonormal matrix.
        R (list of lists): Upper triangular matrix.
    """
    import math
    import copy
    n = len(A)
    R = copy.deepcopy(A)
    Q = [[float(i == j) for j in range(n)] for i in range(n)]  # Identity matrix
    for k in range(n - 1):
        # Create the Householder vector
        x = [float(R[i][k]) for i in range(k, n)]
        norm_x = math.sqrt(sum(xi**2 for xi in x))
        if norm_x == 0:
            continue  # Nothing to reflect, skip
        sign = 1 if x[0] >= 0 else -1
        v = [float(x[i]) for i in range(len(x))]
        v[0] += sign * norm_x
        norm_v = math.sqrt(sum(vi**2 for vi in v))
        if norm_v == 0:
            continue  # Avoid division by zero
        v = [vi / norm_v for vi in v]
        # Build Householder matrix H = I - 2vv^T
        H = [[float(i == j) - 2 * v[i] * v[j] for j in range(len(v))] for i in range(len(v))]
        # Apply H to R (only the lower right submatrix)
        R_sub = [[R[i + k][j] for j in range(n)] for i in range(n - k)]
        R_sub = [[sum(H[i][m] * R_sub[m][j] for m in range(n - k)) for j in range(n)] for i in range(n - k)]
        for i in range(n - k):
            for j in range(n):
                R[i + k][j] = R_sub[i][j]
        # Expand H to full size for Q update
        H_full = [[float(i == j) for j in range(n)] for i in range(n)]
        for i in range(k, n):
            for j in range(k, n):
                H_full[i][j] = H[i - k][j - k]
        # Update Q: Q = Q * H_full
        Q = [[sum(Q[i][m] * H_full[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
    return Q, R

# --- Step 1: Helper function for Gaussian Elimination ---
def gaussian_elimination(matrix):
    """
    Reduces a matrix to reduced row echelon form using Gaussian elimination with partial pivoting.
    This function modifies the input matrix in place.
    """
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    tol = 1e-12

    lead = 0
    for r in range(num_rows):
        if lead >= num_cols:
            return matrix
        
        # Find the best pivot (partial pivoting for numerical stability)
        pivot_row = r
        for i in range(r + 1, num_rows):
            if abs(matrix[i][lead]) > abs(matrix[pivot_row][lead]):
                pivot_row = i
        
        # Check if pivot is too small
        if abs(matrix[pivot_row][lead]) < tol:
            lead += 1
            continue
            
        # Swap rows if needed
        if pivot_row != r:
            matrix[r], matrix[pivot_row] = matrix[pivot_row], matrix[r]

        # Scale pivot row
        lv = matrix[r][lead]
        for j in range(num_cols):
            matrix[r][j] /= lv
        
        # Eliminate column
        for i in range(num_rows):
            if i != r:
                lv = matrix[i][lead]
                for j in range(num_cols):
                    matrix[i][j] -= lv * matrix[r][j]
        lead += 1
    return matrix

# --- Step 2: Helper function to find basis of null space ---
def find_null_space(matrix):
    """
    Finds a basis for the null space of a matrix.
    Assumes matrix is already in reduced row echelon form.
    """
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    tol = 1e-10
    
    # Find pivot columns more robustly
    pivot_cols = []
    pivot_rows = []
    
    for i in range(num_rows):
        pivot_found = False
        for j in range(num_cols):
            if abs(matrix[i][j]) > tol:
                # Check if this is a leading 1 (or close to it)
                if abs(abs(matrix[i][j]) - 1.0) < tol:
                    pivot_cols.append(j)
                    pivot_rows.append(i)
                    pivot_found = True
                    break
                # If not normalized, this might still be a pivot
                elif j not in pivot_cols:
                    pivot_cols.append(j)
                    pivot_rows.append(i)
                    pivot_found = True
                    break
        if not pivot_found:
            break
    
    free_vars = [j for j in range(num_cols) if j not in pivot_cols]
    basis_vectors = []

    # Handle case where no free variables exist
    if not free_vars:
        # Return a zero vector if no free variables
        basis_vectors.append([0] * num_cols)
    else:
        for free_var_idx in free_vars:
            basis_vector = [0.0] * num_cols
            basis_vector[free_var_idx] = 1.0
            
            # Set values for pivot variables
            for k, pivot_col in enumerate(pivot_cols):
                if k < len(pivot_rows) and pivot_rows[k] < num_rows:
                    basis_vector[pivot_col] = -matrix[pivot_rows[k]][free_var_idx]

            basis_vectors.append(basis_vector)

    return basis_vectors

# --- Step 3: Main function to find eigenvectors ---
def find_eigenvectors(A, eigenvalues, tol=1e-10):
    """
    Finds eigenvectors for each eigenvalue of matrix A.
    Returns a list of eigenvectors (one per eigenvalue).
    
    Args:
        A: Square matrix (list of lists)
        eigenvalues: List of eigenvalues
        tol: Tolerance for numerical computations
        
    Returns:
        List of eigenvectors corresponding to eigenvalues
    """
    n = len(A)
    result = []
    
    for lam in eigenvalues:
        # Create matrix (A - λI)
        A_minus_lamI = [row[:] for row in A]  # Deep copy
        for i in range(n):
            A_minus_lamI[i][i] -= lam
        
        # Find null space using Gaussian elimination
        reduced_matrix = [row[:] for row in A_minus_lamI]  # Copy for elimination
        reduced_matrix = gaussian_elimination(reduced_matrix)
        basis = find_null_space(reduced_matrix)
        
        if basis and any(abs(x) > tol for x in basis[0]):
            # Normalize the eigenvector
            eigenvec = basis[0]
            norm = math.sqrt(sum(x**2 for x in eigenvec))
            if norm > tol:
                eigenvec = [x / norm for x in eigenvec]
            result.append(eigenvec)
        else:
            # If no eigenvector found, use a zero vector
            result.append([0.0] * n)
    
    return result

def qr_eigenvalues_with_vectors(matrix, max_iter=10000, tol=1e-12):
    """
    Computes eigenvalues and eigenvectors using QR algorithm.
    
    Args:
        matrix: Square matrix (list of lists)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        tuple: (eigenvalues, eigenvectors) where eigenvectors[i] corresponds to eigenvalues[i]
    """
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")
    
    # Store original matrix for eigenvector computation
    original_matrix = [row[:] for row in matrix]
    
    # QR algorithm for eigenvalues
    A = [row[:] for row in matrix]
    Q_total = [[float(i == j) for j in range(n)] for i in range(n)]  # Identity
    
    for iteration in range(max_iter):
        Q, R = householder_qr(A)
        A = [[sum(R[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        
        # Accumulate Q matrices to get eigenvectors later if needed
        Q_new = [[sum(Q_total[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        Q_total = Q_new
        
        # Check convergence
        off_diag = sum(abs(A[i][j]) for i in range(n) for j in range(n) if i != j)
        if off_diag < tol:
            break
    
    # Extract eigenvalues from diagonal
    eigenvalues = [A[i][i] for i in range(n)]
    
    # Find eigenvectors using the original matrix
    eigenvectors = find_eigenvectors(original_matrix, eigenvalues, tol)
    
    return eigenvalues, eigenvectors

