from matrix_transpose import transpose_matrix
from matrix_det import determinant
from matrix_cofactor import cofactor_matrix

# Certainly! Let’s break down the concept of matrix inversion, the formula, and how your code works, using a simple example.

# What is a Matrix Inverse?
# A matrix inverse is like the reciprocal for numbers. For a square matrix A, its inverse A⁻¹ is another matrix such that:

# A × A⁻¹ = I

# Where I is the identity matrix (like 1 for numbers).

# Formula for Inverse (2x2 Example)
# For a 2x2 matrix:
# A = | a  b |
#     | c  d |
# The inverse is:
# A⁻¹ = (1/(ad - bc)) * |  d  -b |
#                      | -c   a |   
# Here, ad - bc is called the determinant.
# If the determinant is 0, the matrix does not have an inverse (it’s singular).
# Step-by-Step Example
# Suppose we have:
# A = | 4  7 |
#     | 2  6 |
# Determinant:
# det(A) = (4*6) - (7*2) = 24 - 14 = 10
# Since det(A) ≠ 0, A has an inverse.
# Cofactor Matrix:
# For 2x2, the cofactor matrix is:
# |  6  -7 |
# | -2   4 |
# For 2x2, swap diagonal, change sign of off-diagonal:
# |  6   2 |
# | -7   4 |

# Transpose (Adjugate):
# For 2x2, transpose is the same as swapping rows and columns:
# |  6  -7 |
# |  2   4 |
# Inverse Matrix:
# A⁻¹ = (1/10) * |  6  -7 |
#                |  2   4 |
# Divide by Determinant:
# Each element divided by 10:
# A⁻¹ = |  0.6  -0.7 |
#       |  0.2   0.4 |
# How Your Code Works

# Checks if matrix is square
# (must have same number of rows and columns).
# Calculates determinant
# (if zero, matrix can’t be inverted).
# Finds cofactor matrix
# (minor determinants with sign changes).
# Transposes cofactor matrix
# (gets adjugate).
# Divides each element by determinant
# (gets the inverse).
# Summary
# Matrix inverse lets you “undo” the effect of a matrix.
# Formula:
# A⁻¹ = (1/det(A)) × adjugate(A)
# Your code follows this formula step by step.
# Let me know if you want to see the code for determinant, cofactor, or transpose functions!




def matrix_inverse(matrix):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted")
    cofactors = cofactor_matrix(matrix)
    adjugate = transpose_matrix(cofactors)
    inverse = [[adjugate[r][c] / det for c in range(n)] for r in range(n)]
    return inverse

# Example usage:
# mat = [[4, 7], [2, 6]]
# inv = matrix_inverse(mat)
# print(inv)
