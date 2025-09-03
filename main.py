from vector_functions import dot_product
from vector_functions import cross_product
from matrix_multiply import print_matrix, matrix_multiply
from matrix_transpose import transpose_matrix
from matrix_minor import get_minor
from matrix_det import determinant
from matrix_inverse import matrix_inverse

def get_vector_input(prompt):
    return list(map(float, input(prompt).strip().split()))

if __name__ == "__main__":

    print("Choose operation:")
    print("1. Dot Product")
    print("2. Cross Product")
    print("3. Matrix Multiplication")
    print("4. Matrix Transposition")
    print("5. Matrix Inversion")
    print("6. Matrix Minor")
    print("7. Matrix Determinant")
    print("8. Eigenvalues of 2x2 or 3x3 Matrix")
    print("9. Eigenvalues using QR Algorithm (any size)")
    print("10. Singular Value Decomposition (SVD)")
    # print("11. Robust QR Eigenvalues with Diagnostics")
    # print("12. Robust SVD with Options")

    choice = input("Enter 1-12: ").strip()

    if choice == "12":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        
        print("SVD Options:")
        print("1. Full SVD (all matrices)")
        print("2. Economy SVD (compact form)")
        print("3. Singular values only")
        svd_choice = input("Choose (1-3): ").strip()
        
        try:
            from matrix_SVD import svd
            if svd_choice == "3":
                sv = svd(matrix, compute_uv=False)
                print(f"Singular values: {sv}")
            elif svd_choice == "2":
                U, S, Vt = svd(matrix, compute_uv=True, full_matrices=False)
                print("Economy SVD - Shapes:")
                print(f"U: {len(U)}x{len(U[0])}, Sigma: {len(S)}x{len(S[0])}, V^T: {len(Vt)}x{len(Vt[0])}")
                print("U Matrix:")
                print_matrix(U)
                print("Sigma Matrix:")
                print_matrix(S)
                print("V^T Matrix:")
                print_matrix(Vt)
            else:
                U, S, Vt = svd(matrix, compute_uv=True, full_matrices=True)
                print("Full SVD - Shapes:")
                print(f"U: {len(U)}x{len(U[0])}, Sigma: {len(S)}x{len(S[0])}, V^T: {len(Vt)}x{len(Vt[0])}")
                print("U Matrix:")
                print_matrix(U)
                print("Sigma Matrix:")
                print_matrix(S)
                print("V^T Matrix:")
                print_matrix(Vt)
        except Exception as e:
            print(f"Error: {e}")
        exit()
    elif choice == "11":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        try:
            from matrix_qr_eigenvalues import qr_eigenvalues_robust
            result = qr_eigenvalues_robust(matrix)
            print("Robust QR Results:")
            print(f"Eigenvalues: {result['eigenvalues']}")
            print(f"Converged: {result['converged']}")
            print(f"Iterations: {result['iterations']}")
            print(f"Final off-diagonal norm: {result['final_off_diagonal_norm']:.2e}")
            print(f"Condition estimate: {result['condition_estimate']:.2f}")
        except Exception as e:
            print(f"Error: {e}")
        exit()
    elif choice == "10":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        try:
            from matrix_SVD import svd
            U, S, Vt = svd(matrix, verbose=True)
            print("\nSVD Results:")
            print("U Matrix:")
            print_matrix(U)
            print("Sigma Matrix:")
            print_matrix(S)
            print("V^T Matrix:")
            print_matrix(Vt)
        except Exception as e:
            print(f"Error: {e}")
        exit()
    elif choice == "9":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        try:
            from matrix_qr_eigenvalues import qr_eigenvalues
            eigs = qr_eigenvalues(matrix)
            print(f"Eigenvalues: {eigs}")
        except Exception as e:
            print(f"Error: {e}")
        exit()
    elif choice == "8":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        try:
            from matrix_eigenvalues import eigenvalues
            eigs = eigenvalues(matrix)
            if eigs:
                print(f"Eigenvalues: {eigs}")
            else:
                print("No real eigenvalues found.")
        except ImportError:
            print("Eigenvalue function not available.")
        except ValueError as e:
            print(f"Error: {e}")
        exit()
    if choice == "6" or choice == "7":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        if choice == "6":
            print("Enter the row index of the element (0-based):")
            i = int(input().strip())
            print("Enter the column index of the element (0-based):")
            j = int(input().strip())
            minor = get_minor(matrix, i, j)
            print("Minor Matrix:")
            print_matrix(minor)
        elif choice == "7":
            try:
                from matrix_det import determinant
                det = determinant(matrix)
                print(f"Determinant: {det}")
            except ImportError:
                print("Determinant function not available.")
        exit()


    elif choice == "5":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))
        try:
            from matrix_inverse import matrix_inverse
            inv = matrix_inverse(matrix)
            print("Inverse Matrix:")
            print_matrix(inv)
        except ImportError:
            print("Matrix inverse function not available.")
        except ValueError as e:
            print(f"Error: {e}")
        exit()

    if choice == "4":
        print("Enter the matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix.append(list(map(float, line.strip().split())))

        transposed_matrix = transpose_matrix(matrix)
        print("Transposed Matrix:")
        print_matrix(transposed_matrix)         

    elif choice == "3":
        print("Enter the first matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix1 = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix1.append(list(map(float, line.strip().split())))
        print("Enter the second matrix (rows separated by newlines, elements by spaces). End with an empty line:")
        matrix2 = []
        while True:
            line = input()
            if line.strip() == "":
                break
            matrix2.append(list(map(float, line.strip().split())))

        try:
            result_matrix = matrix_multiply(matrix1, matrix2)
            print("Resulting Matrix:")
            print_matrix(result_matrix)
        except ValueError as e:
            print(f"Error: {e}")
    elif choice in ["1", "2"]:
        print("Enter the first vector (space-separated numbers):")
        vector1 = get_vector_input("> ")
        print("Enter the second vector (space-separated numbers):")
        vector2 = get_vector_input("> ")

        if choice == "1":
            if len(vector1) != len(vector2):
                print("Error: Vectors must be of the same length for dot product.")
            else:
                result = dot_product(vector1, vector2)
                print(f"Dot product: {result}")
        elif choice == "2":
            if len(vector1) != 3 or len(vector2) != 3:
                print("Error: Cross product requires 3-dimensional vectors.")
            else:
                result = cross_product(vector1, vector2)
                print(f"Cross product: {result}")
    else:
        print("Invalid choice.")