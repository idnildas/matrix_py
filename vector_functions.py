
# The dot product (also called the scalar product) is a fundamental operation in vector mathematics.

# Mathematical Formula
# Given two vectors of equal length:

# A = [a₁, a₂, ..., aₙ]
# B = [b₁, b₂, ..., bₙ]
# The dot product is calculated as:

# A · B = a₁b₁ + a₂b₂ + ... + aₙbₙ

# Or, using summation notation:

# A · B = Σ (aᵢ × bᵢ) for i = 1 to n

# Example
# If A = [2, 3, 4] and B = [5, 6, 7]:

# A · B = (2×5) + (3×6) + (4×7) = 10 + 18 + 28 = 56

# In Your Code
# The code uses zip(vec1, vec2) to pair elements, multiplies each pair, and sums the results—directly implementing the formula above.

# Gotcha
# Both vectors must be the same length, or the operation is undefined (hence the ValueError).
# Let me know if you want a geometric interpretation or more examples!



def dot_product(vec1, vec2):
    """
    Returns the dot product of two vectors.
    Args:
        vec1 (list or tuple): First vector.
        vec2 (list or tuple): Second vector.
    Returns:
        float or int: Dot product of the vectors.
    Raises:
        ValueError: If vectors are not the same length.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length.")
    return sum(a * b for a, b in zip(vec1, vec2))


# similarly, the cross product is a binary operation on two vectors in three-dimensional space.
# It produces a third vector that is orthogonal to the plane formed by the input vectors.
# The cross product of two 3-dimensional vectors a and b produces a new vector that is perpendicular to both. If
# a = [a₁, a₂, a₃]
# b = [b₁, b₂, b₃]

# The cross product a × b is calculated as:

# x-component: a₂·b₃ − a₃·b₂
# y-component: a₃·b₁ − a₁·b₃
# z-component: a₁·b₂ − a₂·b₁
# So, the result is:
# [a₂·b₃ − a₃·b₂, a₃·b₁ − a₁·b₃, a₁·b₂ − a₂·b₁]

# This matches the formulas in your code. 
# The cross product is widely used in physics and engineering to find a vector orthogonal to two given vectors, 
# such as for torque or rotational motion.




def cross_product(vec1, vec2):
    """
    Returns the cross product of two 3-dimensional vectors.
    Args:
        vec1 (list or tuple): First vector (length 3).
        vec2 (list or tuple): Second vector (length 3).
    Returns:
        list: Cross product vector.
    Raises:
        ValueError: If vectors are not both length 3.
    """
    if len(vec1) != 3 or len(vec2) != 3:
        raise ValueError("Both vectors must be of length 3 for cross product.")
    return [
        vec1[1]*vec2[2] - vec1[2]*vec2[1],
        vec1[2]*vec2[0] - vec1[0]*vec2[2],
        vec1[0]*vec2[1] - vec1[1]*vec2[0]
    ]