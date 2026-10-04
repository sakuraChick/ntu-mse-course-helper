import numpy as np
import pandas as pd

class Sample:
    def __init__(self, label: str, file_path: str, thickness: np.float32, length: np.float32, width: np.float32):
        self.label = label
        self.thickness = thickness
        self.length = length
        self.width = width

        df = pd.read_csv(file_path, header=0)

        force = np.array(df.get('Force'), dtype=np.float32)
        displacement = np.array(df.get('Displacement'), dtype=np.float32)
        self.time = np.array(df.get('Time'), dtype=float)

        self.stress = force * 1e3 / self.area() * 1e-6 # in MPa
        self.strain = displacement / self.length

    def area(self) -> np.float32:
        """
        :return: the cross-section area in m^2
        """
        return self.thickness * self.width * 1e-6

    def __repr__(self) -> str:
        return f"Sample {self.label}"