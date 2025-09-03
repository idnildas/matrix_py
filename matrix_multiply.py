# matrix_multiply.py
# layman_explanation:
    # This function multiplies two matrices together. A matrix is a rectangular array of numbers arranged
    # in rows and columns. The multiplication of two matrices is only possible when the number of columns
    # in the first matrix is equal to the number of rows in the second matrix.  
    # The resulting matrix has a number of rows equal to the first matrix and a number of columns equal to the second matrix.
    # Each element in the resulting matrix is calculated by taking the dot product of the corresponding row
    # from the first matrix and the corresponding column from the second matrix.
    # show graphically how the multiplication works
    # For example, if we have:
    #     A = [[1, 2, 3],
    #          [4, 5, 6]]
    #     B = [[7, 8],
    #          [9, 10],
    #          [11, 12]]        
    # The resulting matrix C will be:
    #     C = [[(1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12)],   
    #          [(4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12)]]
    #     C = [[58, 64],
    #          [139, 154]]
    # Example:
    # Given:
    #     - Matrix A of size m x n
    #     - Matrix B of size n x p

    #     - Matrix C of size m x p, where each element C[i][j] is computed as:
    #         C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))

    #     ValueError: If the number of columns in A does not match the number of rows in B.

    # Mathematical Explanation:
    #     Matrix multiplication is defined such that for matrices A (m x n) and B (n x p),
    #     the resulting matrix C (m x p) has elements:
    #         C[i][j] = Σ (A[i][k] * B[k][j]) for k = 0 to n-1
    #     This operation is only valid when the number of columns in A equals the number of rows in B.

def matrix_multiply(A, B):
    """
    Multiplies two matrices A and B.
    Args:
        A (list of list of numbers): Matrix of size m x n
        B (list of list of numbers): Matrix of size n x p
    Returns:
        list of list of numbers: Resulting matrix of size m x p
    Raises:
        ValueError: If matrices cannot be multiplied due to incompatible dimensions.
    """
    if not A or not B or len(A[0]) != len(B):
        raise ValueError("Incompatible matrix dimensions for multiplication.")

    m, n = len(A), len(A[0])
    p = len(B[0])
    result = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result



def print_matrix(matrix):
    """
    Prints the matrix in a readable, formatted way.
    Args:
        matrix (list of list of numbers): The matrix to print.
    """
    for row in matrix:
        print("[" + " ".join(f"{elem:.2f}" for elem in row) + "]")