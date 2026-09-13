import numpy as np
from utils import to_ref

def main():
    A = np.array([[1, 2, 3, 8],
                  [4, 5, 6, 6],
                  [7, 8, 9, 0]])

    print(to_ref(A))

if __name__ == '__main__':
    main()