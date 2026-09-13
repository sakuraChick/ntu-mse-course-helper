import math
import scipy.constants as constants
from scipy.special import erfc, erfcinv
from utils import boltzmann_distribution_for_A, boltzmann_distribution_for_E, boltzmann_distribution_for_T


def bragg_equation(wave_length: float,
              theta: float,
              n: int,
              h: float,
              k: float,
              l: float) -> float:
    """
    About Bragg's law: https://en.wikipedia.org/wiki/Bragg%27s_law.
    Note that in questions usually 2θ is given.
    :param wave_length: in nm
    :param theta: the glancing angle in radians
    :param n: diffraction order
    :param h: first in Miller indices
    :param k: second in Miller indices
    :param l: third in Miller indices
    :return: the lattice parameter in nm
    """
    d_hkl = n * wave_length / (2 * math.sin(theta))
    a = d_hkl * math.sqrt(h ** 2 + k ** 2 + l ** 2)
    return a

def arrhenius_equation_for_n(N: float, Q_v: float, T: float) -> float:
    """
    About Arrhenius Law: https://en.wikipedia.org/wiki/Arrhenius_equation.
    :param N: in atoms / m^-3
    :param Q_v: in J
    :param T: in K
    :return: the vacancy in lattice in atoms / m^-3
    """
    k_b = constants.k
    return boltzmann_distribution_for_A(N, Q_v, k_b, T)

def arrhenius_equation_for_Qv(n: float,
                              N: float,
                              T: float) -> float:
    """
    About Arrhenius Law: https://en.wikipedia.org/wiki/Arrhenius_equation.
    :param n: in atoms / m^-3
    :param N: in atoms / m^-3
    :param T: in K
    :return: activation energy in J
    """
    k_b = constants.k
    return boltzmann_distribution_for_E(n, N, k_b, T)

def arrhenius_equation_for_T(n: float,
                             N: float,
                             Q_v: float) -> float:
    """
    About Arrhenius Law: https://en.wikipedia.org/wiki/Arrhenius_equation.
    :param n: in atoms / m^-3
    :param N: in atoms / m^-3
    :param Q_v: in J
    :return: current temperature in K
    """
    k_b = constants.k
    return boltzmann_distribution_for_T(n, N, Q_v, k_b)

def fick_2_law_for_C_xt(C_0: float,
                        C_s: float,
                        x: float,
                        D: float,
                        t: float) -> float:
    """
    About Fick's Law of diffusion: https://en.wikipedia.org/wiki/Fick%27s_laws_of_diffusion.
    This is only for constant surface concentration model!
    :param C_0: C(x, 0)
    :param C_s: C(0, t) (surface concentration)
    :param x: position in m
    :param D: diffusion constant in m^2 * s^-1
    :param t: time in s
    :return: C(x, t)
    """
    z = x / math.sqrt(2 * D * t)
    err = erfc(z)
    C_xt = C_0 + (C_s - C_0) * err
    return C_xt

def fick_2_law_for_D(C_xt: float,
                        C_0: float,
                        C_s: float,
                        x: float,
                        t: float) -> float:
    """
    About Fick's Law of diffusion: https://en.wikipedia.org/wiki/Fick%27s_laws_of_diffusion.
    This is only for constant surface concentration model!
    :param C_xt: C(x, t)
    :param C_0: C(x, 0)
    :param C_s: C(0, t) (surface concentration)
    :param x: position in m
    :param t: time in s
    :return: diffusion constant in m^2 * s^-1
    """
    y = (C_xt - C_0) / (C_s - C_0)
    z = erfcinv(y)
    D = x ** 2 / (4 * (z ** 2) * t)
    return D

