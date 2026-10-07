from dataclasses import dataclass
import numpy as np


@dataclass
class YieldResult:
    label: str
    E: float
    m: float
    r2: float
    start_idx: int
    end_idx: int

def linear_generator_with_offset(E: np.float32, m: np.float32):
    OFFSET = 0.002
    return lambda x_hat: E * (x_hat - OFFSET) + m