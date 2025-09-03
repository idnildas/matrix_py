
# The transpose of a matrix is obtained by swapping its rows with its columns.
# For a matrix A of size m x n, the transpose A^T will be of size n x m, where
# the element at position (i, j) in A becomes the element at position (j, i) in A^T.
# For example, if we have:
# A = [[1, 2, 3],
#      [4, 5, 6]]
# The transpose A^T will be:
# A^T = [[1, 4],
#        [2, 5],
#        [3, 6]]

def transpose_matrix(matrix):
    """
    Returns the transpose of the given matrix.

    Args:
        matrix (list of list of numbers): The input matrix.

    Returns:
        list of list of numbers: The transposed matrix.
    """
    if not matrix:
        return []

    # Transpose using zip and list comprehension
    # return [list(row) for row in zip(*matrix)] # this is built-in way
    transposed = []
    rows = len(matrix)
    cols = len(matrix[0])   
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        transposed.append(new_row)
    return transposed
