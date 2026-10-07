import numpy as np
from scipy.stats import linregress
from scipy.integrate import trapezoid

from .sample import Sample
from .lab_utils import linear_generator_with_offset
from .lab_utils import YieldResult


class DataProcessor:
    # @staticmethod
    # def analyze_yield(s: Sample) -> YieldResult:
    #     """
    #     :param s: The sample
    #     :return: yield_strain, yield_stress, k, m, yield_index
    #     """
    #     ERROR = 0.0002
    #     E = np.float32(0)
    #     E_max = np.float32(0)
    #     m = np.float32(0)
    #     Es = []
    #     sigmas = []
    #     yield_stress = np.float32(0)
    #     yield_strain = np.float32(0)
    #
    #     yield_index = 0
    #
    #     print(
    #         f"Starting analyzing {s}"
    #     )
    #     print("================================================")
    #
    #     for i, time in enumerate(s.time):
    #         r = linregress(s.strain[:i], s.stress[:i])
    #
    #         print(
    #             f"Time: {time} | strain: {s.strain[i]:<5f} | E: {r.slope:<5f}, m: {r.intercept:<5f} | r: {r.rvalue:<3f}"
    #         )
    #
    #         Es.append(r.slope)
    #         sigmas.append(s.strain[i])
    #
    #         if r.slope > E_max:
    #             E_max = r.slope
    #             continue
    #
    #         if r.slope < (1 - ERROR) * E_max: # check if young's modulus is askew from a straight line.
    #             E = np.float32(r.slope)
    #             m = np.float32(r.intercept)
    #             break
    #
    #     for i, _ in enumerate(s.time):
    #         if abs(s.stress[i] - linear_generator_with_offset(E, m)(s.strain[i])) < 1:
    #             yield_strain = s.strain[i]
    #             yield_stress = s.stress[i]
    #             yield_index = i
    #             break
    #
    #     return YieldResult(str(s), E, m, yield_strain, yield_stress, yield_index)

    @staticmethod
    def find_modulus(s: Sample, win=30, lo=0.1, hi=0.6):
        s_max = np.max(s.stress)
        rg = np.where((s.stress > lo * s_max) & (s.stress < hi * s_max))
        e, s = s.strain[rg], s.stress[rg]
        n = len(e)

        result = None
        for i in range(0, n - win):
            x, y = e[i:i + win], s[i:i + win]
            r = linregress(x, y)
            if result is None or (r.rvalue > 0.999 and r.slope > result.E):
                result = YieldResult(str(s), r.slope, r.intercept, r.rvalue, i, i+win)
        return result

    @staticmethod
    def sliding_slopes(s: Sample, win=30, lo=0.1, hi=0.6):
        smax = np.max(s.stress)
        i_max = np.argmax(s.stress)
        rg = np.where((s.stress[:i_max+1] > lo * smax) & (s.stress[:i_max+1] < hi * smax))[0]
        e, s = s.strain[rg], s.stress[rg]

        out = []
        for i in range(0, len(e) - win):
            x, y = e[i:i + win], s[i:i + win]
            r = linregress(x, y)
            out.append((x.mean(), r.slope, r.rvalue, rg[i], rg[i + win - 1]))
        return np.array(out), e, s

    @staticmethod
    def find_yield(s: Sample, result: YieldResult):
        yield_strain = None
        yield_stress = None
        yield_index = None
        for i, _ in enumerate(s.time):
            if abs(s.stress[i] - linear_generator_with_offset(result.E, result.m)(s.strain[i])) < 1:
                yield_strain = s.strain[i]
                yield_stress = s.stress[i]
                yield_index = i

        return yield_strain, yield_stress, yield_index

    @staticmethod
    def analyze_tensile_strength(s: Sample) -> np.float32:
        return np.max(s.stress)

    @staticmethod
    def analyze_fracture_stress(s: Sample) -> np.float32:
        return s.stress[-5]

    @staticmethod
    def analyze_ductility(s: Sample) -> np.float32:
        """In percentage elongation (%EL)
        """
        return s.strain[-1] * 100

    @staticmethod
    def analyze_toughness(s: Sample) -> np.float32:
        return trapezoid(s.stress, s.strain)

    @staticmethod
    def analyze_all(a: Sample, *args: Sample):
        for s in (a, *args):
            yield DataProcessor.analyze_yield(s)