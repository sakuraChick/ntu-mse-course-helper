import numpy as np

from ms1017.lab.data_plotter import DataPlotter
from ms1017.lab.data_processor import DataProcessor
from ms1017.lab.sample import Sample


def main():

    a = Sample('A', "./data/sample_a.csv", np.float32(1.32), np.float32(76.2), np.float32(3.14))
    b = Sample('B', "./data/sample_b.csv", np.float32(1.94), np.float32(76.2), np.float32(3.15))
    c = Sample('C', "./data/sample_c.csv", np.float32(1.19), np.float32(76.2), np.float32(3.14))
    d = Sample('D', "./data/sample_d.csv", np.float32(1), np.float32(76.2), np.float32(3.17))

    res, e, s= DataProcessor.sliding_slopes(c, lo=0.0)
    result = DataProcessor.find_modulus(c, lo=0)
    yield_e, yield_s, yield_i = DataProcessor.find_yield(c, result)
    print(result.E)
    DataPlotter.plot_check(e, s, res)
    DataPlotter.plot_in_yield(d, result.E, result.m, yield_i+20, yield_e, yield_s)

if __name__ == '__main__':
    main()