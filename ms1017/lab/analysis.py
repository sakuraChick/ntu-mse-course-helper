from .sample import Sample
from .data_processor import DataProcessor
from .data_plotter import DataPlotter


def analyze_specific(s: Sample) -> str:
    pack = DataProcessor.analyze_yield(s)
    DataPlotter.plot_in_yield(s, pack.E, pack.m, pack.yield_index+10, pack.yield_strain, pack.yield_stress)

    return (f"========================================\n"
            f"Sample {s.label}:\n"
            f"----------------------------------------\n"
            f"Basic Information:\n"
            f"\tlength: {s.length:2f} mm\n"
            f"\twidth: {s.width:2f} mm\n"
            f"\tthickness: {s.thickness:2f} mm\n"
            f"----------------------------------------\n"
            f"Yield Performance:\n"
            f"\tYoung's modulus: {pack.E * 0.001:3f} GPa\n"
            f"\tyield stress: {pack.yield_stress:3f} MPa\n"
            f"----------------------------------------\n"
            f"Limit Performance:\n"
            f"\ttensile strength: {DataProcessor.analyze_tensile_strength(s):3f} MPa\n"
            f"\tfracture strength: {DataProcessor.analyze_fracture_stress(s):3f} MPa\n"
            f"\tductility: {DataProcessor.analyze_ductility(s):5f}%\n"
            f"\ttoughness: {DataProcessor.analyze_toughness(s):6f} MPa\n"
            f"========================================")


def analyze_general(s: Sample, *args: Sample):
    DataPlotter.plot_stress_to_strain(s, *args)