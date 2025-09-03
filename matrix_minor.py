
# Get the minor of a matrix
# Certainly!
# The get_minor function helps you find the minor of a matrix element, which is a key step in calculating a matrix's determinant or inverse.

# What is a Minor?
# For a given element in a matrix (say, at row i and column j), its minor is the determinant of the smaller matrix you get by removing that row and column.

# Formula:
# If A is an n x n matrix, the minor of element A[i][j] is the determinant of the matrix formed by deleting row i and column j from A.

# Example
# Suppose you have a 3x3 matrix:
# A = | 1  2  3 |
#     | 4  5  6 |
#     | 7  8  9 |

# Let's find the minor for the element at row 1, column 1 (A[1][1], which is 5):

# Remove row 1 ([4, 5, 6])
# Remove column 1 (the second column)
# The remaining matrix is:
# | 1  3 |
# | 7  9 |
# The minor of A[1][1] is the determinant of this 2x2 matrix:
# Minor = (1*9) - (3*7) = 9 - 21 = -12  
# How the Code Works
# The function get_minor(matrix, i, j) constructs this smaller matrix.
# It uses a list comprehension to iterate through each row of the original matrix.

# enumerate(matrix) loops through each row with its index.
# if idx != i skips the row at index i.
# row[:j] + row[j+1:] removes the column at index j from each row.
# Using the Example
# Output:
# minor = get_minor(A, 1, 1)
# print(minor)  # Output: [[1, 3], [7, 9]]
# This will give you the 2x2 matrix that is the minor of the element at row 1, column 1.
# Summary
# Minor = matrix after removing the specified row and column.
# Used for calculating determinants and inverses.
# The code efficiently creates this smaller matrix for any element.

def get_minor(matrix, i, j):
    # return [row[:j] + row[j+1:] for idx, row in enumerate(matrix) if idx != i]
    minor = []
    for idx in range(len(matrix)):
        if idx != i:
            row = []
            for col in range(len(matrix[idx])):
                if col != j:
                    row.append(matrix[idx][col])
            minor.append(row)
    return minor


