from matrix_minor import get_minor

# The determinant of a matrix is a scalar value that can be computed from the elements of a square matrix.
# It is useful in linear algebra for solving systems of equations, finding inverses, and understanding matrix properties.
# For a 2x2 matrix:
# |a b|
# |c d|
# The determinant is calculated as: det = a*d - b*c
# For larger matrices, the determinant is calculated recursively using minors and cofactors.
# Example for a 3x3 matrix:
# |a b c|
# |d e f|
# |g h i|
# det = a * det(minor of a) - b * det(minor of b) + c * det(minor of c)
# where the minor is the determinant of the submatrix formed by removing the row and column of the element.

def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        det += ((-1)**c) * matrix[0][c] * determinant(get_minor(matrix, 0, c))
    return det