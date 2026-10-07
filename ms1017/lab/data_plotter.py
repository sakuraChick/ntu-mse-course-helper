import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .sample import Sample
from .lab_utils import linear_generator_with_offset


class DataPlotter:
    @staticmethod
    def plot_stress_to_strain(a: Sample, *args: Sample):
        sample_list = [a, *args]
        colors = ['red', 'blue', 'green', 'purple']

        plt.figure(figsize=(8, 5))

        for i, s in enumerate(sample_list):
            plt.plot(s.strain, s.stress, color=colors[i%len(colors)], label=f"Sample {chr(65+i)}")

        plt.title("Stress-strain graph")

        plt.xlabel("Strain")
        plt.ylabel("Stress / MPa")

        plt.tight_layout()
        plt.legend()
        plt.show()

    @staticmethod
    def plot_in_yield(s: Sample, E: np.float32, m: np.float32, yield_index: int, yield_strain: np.float32, yield_stress: np.float32):
        x_hat = np.linspace(0.002, s.strain[yield_index], 100)
        y_hat = linear_generator_with_offset(E, m)(x_hat)

        plt.figure(figsize=(5, 8))

        plt.plot(x_hat, y_hat, color="red", linewidth=2, label='fit line')
        plt.plot(s.strain[:yield_index+200], s.stress[:yield_index+200], color='blue', label=str(s))

        plt.scatter(yield_strain, yield_stress, s=20, color="black")
        plt.annotate(
            f"({yield_strain}, {yield_stress})",
            (yield_strain, yield_stress),
            xytext=(-125, 10),
            textcoords="offset points"
        )

        plt.title("Sample within yield strain (with fit line)")
        plt.xlabel("Strain")
        plt.ylabel("Stress / MPa")

        plt.tight_layout()
        plt.legend()
        plt.show()

    @staticmethod
    def plot_check(strain, stress, res):
        fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        ax[0].plot(strain, stress, lw=1)
        ax[0].set_ylabel("Stress (MPa)")
        ax[1].plot(res[:, 0], res[:, 1], "-", ms=3)
        ax[1].set_ylabel("Window slope (E, MPa)")
        ax[2].plot(res[:, 0], res[:, 2], "-", ms=3)
        ax[2].set_ylabel("R²")
        ax[2].set_xlabel("Strain")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def merge_e2sigma():
        pass