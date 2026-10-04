import numpy as np

from ms1017.lab.sample import Sample
from ms1017.lab.analysis import analyze_specific
from ms1017.lab.analysis import analyze_general


def main():

    a = Sample('A', "./data/sample_a.csv", np.float32(1.32), np.float32(76.2), np.float32(3.14))
    b = Sample('B', "./data/sample_b.csv", np.float32(1.94), np.float32(76.2), np.float32(3.15))
    c = Sample('C', "./data/sample_c.csv", np.float32(1.19), np.float32(76.2), np.float32(3.14))
    d = Sample('D', "./data/sample_d.csv", np.float32(1), np.float32(76.2), np.float32(3.17))

    print(analyze_specific(d))

if __name__ == '__main__':
    main()