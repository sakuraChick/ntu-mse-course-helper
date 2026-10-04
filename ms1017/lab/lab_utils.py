from dataclasses import dataclass
import numpy as np


@dataclass
class YieldDataPack:
    label: str
    E: np.float32
    m: np.float32
    yield_strain: np.float32
    yield_stress: np.float32
    yield_index: int

def linear_generator_with_offset(E: np.float32, m: np.float32):
    OFFSET = 0.002
    return lambda x_hat: E * (x_hat - OFFSET) + m