import math
import numpy as np
import numpy.typing as npt

def boltzmann_distribution_for_A(A_0: float,
                                 E: float,
                                 const: float,
                                 T: float) -> float:
    """
    A = A_0 * exp(-E / (const * T))
    About Boltzmann distribution: https://en.wikipedia.org/wiki/Boltzmann_distribution
    :param A_0: Pre-exponential constant
    :param E: J * mol^-1 if const is R, J if const is k_b
    :param const: may be R or k_b
    :param T: usually temperature in K
    :return: the energy, same unit with A_0
    """
    return A_0 * math.exp(-E / (const * T))

def boltzmann_distribution_for_E(A: float,
                                 A_0: float,
                                 const: float,
                                 T: float) -> float:
    """
    A = A_0 * exp(-E / (const * T))
    About Boltzmann distribution: https://en.wikipedia.org/wiki/Boltzmann_distribution
    :param A: same unit with A_0
    :param A_0: Pre-exponential constant
    :param const: may be R or k_b
    :param T: usually temperature in K
    :return: energy barrier in J * mol^-1 if const is R, J if const is k_b
    """
    return -const * T * math.log(A / A_0)

def boltzmann_distribution_for_T(A: float,
                                 A_0: float,
                                 E: float,
                                 const: float) -> float:
    """
    A = A_0 * exp(-E / (const * T))
    About Boltzmann distribution: https://en.wikipedia.org/wiki/Boltzmann_distribution
    :param A: Pre-exponential constant
    :param A_0: Pre-exponential constant
    :param E: J * mol^-1 if const is R, J if const is k_b
    :param const: may be R or k_b
    :return: current temperature in K
    """
    return -E / (const * math.log(A / A_0))

def to_ref(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    :param A: a matrix
    :return: the matrix in row echelon form
    """
    m, n = A.shape
    pivot_row = 0

    for col in range(n):
        if A[pivot_row, col] == 0:
            non_zero_row = _search_none_zero(A, m, pivot_row, col)
            if non_zero_row is not None:
                _swap(A, pivot_row, non_zero_row)
            else:
                continue

        for row in range(pivot_row+1, m):
            _eliminate(A, pivot_row, row, col)
        pivot_row += 1

    return A

def to_rref(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    pass

def _swap(A: npt.NDArray[np.float64], k: int, l: int) -> None:
    """
    :param A: the matrix
    :param k: the zero row
    :param l: the non-zero row
    """
    A[k], A[l] = A[l], A[k]

def _eliminate(A: npt.NDArray[np.float64], pivot_row: int, to_be_eliminated: int, col: int) -> None:
    """
    :param A: the matrix
    :param pivot_row: the pivot row
    :param to_be_eliminated: to be eliminated
    :param col: which column to eliminate
    """
    A[to_be_eliminated] = A[to_be_eliminated] - (A[pivot_row] / A[pivot_row, col]) * A[to_be_eliminated, col]

def _search_none_zero(A: npt.NDArray[np.float64], m: int, row: int, col: int) -> int | None:
    """
    :param A: the matrix
    :param m: maximum search depth
    :param row: initial row
    :param col: the column
    :return: the index of first non-zero row / nothing (free variable)
    """
    for i in range(row, m):
        if A[i, col] != 0:
            return i

    return None

