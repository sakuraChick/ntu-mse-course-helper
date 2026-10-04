import numpy as np
from scipy.stats import linregress
from scipy.integrate import trapezoid

from .sample import Sample
from .lab_utils import linear_generator_with_offset
from .lab_utils import YieldDataPack


class DataProcessor:
    @staticmethod
    def analyze_yield(s: Sample) -> YieldDataPack:
        """
        :param s: The sample
        :return: yield_strain, yield_stress, k, m, yield_index
        """
        ERROR = 0.0002
        E = np.float32(0)
        E_max = np.float32(0)
        m = np.float32(0)
        Es = []
        sigmas = []
        yield_stress = np.float32(0)
        yield_strain = np.float32(0)

        yield_index = 0

        print(
            f"Starting analyzing {s}"
        )
        print("================================================")

        for i, time in enumerate(s.time):
            r = linregress(s.strain[:i], s.stress[:i])

            print(
                f"Time: {time} | strain: {s.strain[i]:<5f} | E: {r.slope:<5f}, m: {r.intercept:<5f} | r: {r.rvalue:<3f}"
            )

            Es.append(r.slope)
            sigmas.append(s.strain[i])

            if r.slope > E_max:
                E_max = r.slope
                continue

            if r.slope < (1 - ERROR) * E_max:
                E = np.float32(r.slope)
                m = np.float32(r.intercept)
                break

        for i, _ in enumerate(s.time):
            if abs(s.stress[i] - linear_generator_with_offset(E, m)(s.strain[i])) < 1:
                yield_strain = s.strain[i]
                yield_stress = s.stress[i]
                yield_index = i
                break

        return YieldDataPack(str(s), E, m, yield_strain, yield_stress, yield_index)

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