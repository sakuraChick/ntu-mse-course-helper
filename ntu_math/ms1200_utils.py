import numpy as np
import numpy.typing as npt

def projection(A: npt.NDArray, b: npt.NDArray):
    P = A @ np.linalg.inv(A.T @ A) @ A.T
    x_hat = np.linalg.inv(A.T @ A) @ A.T @ b
    return P, x_hat

def main():
    A = np.array([[1, -2],
                  [1, -1],
                  [1, 0],
                  [1, 1],
                  [1, 2]])
    b = np.array([[4],
                  [2],
                  [-1],
                  [0],
                  [0]])

    P, x_hat = projection(A, b)
    print(P, x_hat)

if __name__ == '__main__':
    main()