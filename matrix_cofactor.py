from matrix_minor import get_minor
from matrix_det import determinant

# This module provides functionality to compute the cofactor matrix of a given square matrix.

# A cofactor matrix is a matrix where each element is the cofactor of the corresponding element in the original matrix.
# The cofactor of an element a_ij in a matrix is calculated as:
#     C_ij = (-1)^(i+j) * det(M_ij)
# where:
#     - i, j are the row and column indices of the element,
#     - M_ij is the minor of the element (the matrix formed by removing the i-th row and j-th column),
#     - det(M_ij) is the determinant of the minor.

# Example:
# Given a 2x2 matrix:
#     | a b |
#     | c d |

# The cofactor matrix is:
#     |  det([[d]])   -det([[c]]) |
#     | -det([[b]])    det([[a]]) |

# For a 3x3 matrix:
#     | a b c |
#     | d e f |
#     | g h i |

# The cofactor of element at (0, 0) is:
#     C_00 = (-1)^(0+0) * det([[e, f], [h, i]])
# and similarly for other elements.

# This module assumes the existence of:
#     - get_minor(matrix, row, col): Returns the minor matrix after removing the specified row and column.
#     - determinant(matrix): Returns the determinant of the given matrix.

# Use this module to compute the cofactor matrix, which is useful in finding the adjugate and inverse of a matrix.



def cofactor_matrix(matrix):
    n = len(matrix)
    cofactors = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            minor = get_minor(matrix, r, c)
            cofactor = ((-1) ** (r + c)) * determinant(minor)
            cofactor_row.append(cofactor)
        cofactors.append(cofactor_row)
    return cofactors