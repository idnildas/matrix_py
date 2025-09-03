import math

# Function to compute eigenvalues of a square matrix (2x2 or 3x3) without any libraries
# For 2x2: Uses quadratic formula
# For 3x3: Uses cubic formula (Cardano's method)
# For larger matrices, this function does not support

# Eigenvalues are special numbers associated with a square matrix.
# For a matrix A, an eigenvalue λ satisfies: A*v = λ*v for some nonzero vector v.
# To find eigenvalues, we solve the "characteristic equation":
#     det(A - λI) = 0
# For a 2x2 matrix [[a, b], [c, d]]:
#     The characteristic equation is:
#         |a-λ  b  |
#         |c   d-λ| = 0
#     Expanding gives:
#         (a-λ)(d-λ) - b*c = 0
#         λ^2 - (a+d)λ + (ad-bc) = 0
#     This is a quadratic equation in λ.
#     Example:
#         For matrix [[4, 2], [1, 3]]:
#             a=4, b=2, c=1, d=3
#             Characteristic equation: λ^2 - 7λ + 10 = 0
#             Solutions: λ = 5, λ = 2
# For a 3x3 matrix, the characteristic equation is cubic:
#     λ^3 + aλ^2 + bλ + c = 0
#     Coefficients a, b, c are calculated from the matrix entries.
#     Solving a cubic equation is more complex (Cardano's method).
#     Example:
#         For matrix [[1, 2, 3], [0, 4, 5], [0, 0, 6]]:
#             The eigenvalues are 1, 4, 6 (since it's upper triangular).

def eigenvalues(matrix):
    """
    Returns the eigenvalues of a 2x2 or 3x3 matrix.
    Args:
        matrix: List of lists representing a square matrix.
    Returns:
        List of eigenvalues (real only).
    """
    n = len(matrix)
    # Check if matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")

    if n == 2:
        # For 2x2 matrix [[a, b], [c, d]]
        a, b = matrix[0]
        c, d = matrix[1]
        # Characteristic equation: λ^2 - (a+d)λ + (ad-bc) = 0
        trace = a + d
        det = a * d - b * c
        # Quadratic formula
        discriminant = trace**2 - 4 * det
        if discriminant < 0:
            # Only real eigenvalues are returned
            return []
        sqrt_disc = discriminant ** 0.5
        eig1 = (trace + sqrt_disc) / 2
        eig2 = (trace - sqrt_disc) / 2
        return [eig1, eig2]

    elif n == 3:
        # For 3x3 matrix, use cubic formula
        # Characteristic equation: λ^3 + jλ^2 + lλ + m = 0
        # where: j = -(a+e+i), l = (ae + ai + ei - bd - cg - fh), m = -(aei + bfg + cdh - afh - bdi - ceg)
        # Matrix elements
        # | a  b  c |
        # | d  e  f |
        # | g  h  i |
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[0][2]
        d = matrix[1][0]
        e = matrix[1][1]
        f = matrix[1][2]
        g = matrix[2][0]
        h = matrix[2][1]
        i = matrix[2][2]
        print("Matrix elements:", a, b, c, d, e, f, g, h, i)
        # now we reduce the equation to the form x^3 + px + q = 0
        # where p and q are calculated as follows:
        # before reduction we need j, l, m which is actually b' , c' and d' in the simple form and a' is 1
        j = -(a + e + i)
        l = (a*e + a*i + e*i - b*d - c*g - f*h)
        m = -(a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g)
        print("a', b', c', d':",1, j, l, m)
        # p = (3*l - j**2) / 3
        # q = (2*(j**3) - 9*j*l + 27*m) / 27
        p = (3*l - j**2) / 3
        q = (2*(j**3) - 9*j*l + 27*m) / 27
        print("p:", p, "q:", q)
        # Discriminant
        discriminant = (q/2)**2 + (p/3)**3
        print("Discriminant:", discriminant)
        eigenvalues = []
        if discriminant > 0:
            # One real root
            print("complex roots not supported yet")
        elif discriminant == 0:
            # All roots real, at least two are equal
            # the roots are as follows:
            # y1 = y2 = -3*q/(2*p)
            # y3 = -(-q/2)^(1/3)
            # if p = q = 0 then all roots are zero
            if p == 0 and q == 0:
                return [0, 0, 0]
            y1 = -3 * q / (2 * p)
            y3 = -(-q / 2) ** (1 / 3)
            print("y1, y3:", y1, y3)
            #x = y - (a + e + i) / 3
            x1 = y1 - (a + e + i) / 3
            x3 = y3 - (a + e + i) / 3
            print("x1, x3:", x1, x3)
            eigenvalues = [x1, x1, x3]
            return eigenvalues
        elif discriminant < 0:
            # Three distinct real roots
            # we need to compute r and phi
            # r = 2*sqrt(-p/3)
            # phi = acos(3*q/(2*p)*sqrt(-3/p))
            r = 2 * math.sqrt(-p / 3)
            phi = math.acos(3 * q / (2 * p) * math.sqrt(-3 / p))
            print("r, phi:", r, phi)
            # the three roots are given by:
            # yk = r*cos((phi + 2*k*pi)/3) for k = 0, 1, 2
            # then xk = yk - (a + e + i) / 3
            for k in range(3):
                # Compute yk
                yk = round(r * math.cos((phi + 2 * k * math.pi) / 3), 2)
                print("y%f:", k, yk)
                xk = yk - j / 3
                print("x%f:", k, xk)
                eigenvalues.append(xk)
            return eigenvalues
        else:
            raise ValueError("Unexpected case in discriminant evaluation")
    else:
        raise NotImplementedError("Eigenvalue calculation only implemented for 2x2 and 3x3 matrices")


