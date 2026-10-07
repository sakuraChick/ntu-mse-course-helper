from .sample import Sample
from .data_processor import DataProcessor
from .data_plotter import DataPlotter


def analyze_specific(s: Sample) -> str:
    res, e, strain = DataProcessor.sliding_slopes(s, lo=0.0)
    result = DataProcessor.find_modulus(s, lo=0)
    yield_e, yield_s, yield_i = DataProcessor.find_yield(s, result)
    DataPlotter.plot_check(e, strain, res)
    DataPlotter.plot_in_yield(s, result.E, result.m, yield_i + 20, yield_e, yield_s)

    return (f"========================================\n"
            f"Sample {s.label}:\n"
            f"----------------------------------------\n"
            f"Basic Information:\n"
            f"\tlength: {s.length:2f} mm\n"
            f"\twidth: {s.width:2f} mm\n"
            f"\tthickness: {s.thickness:2f} mm\n"
            f"----------------------------------------\n"
            f"Yield Performance:\n"
            f"\tYoung's modulus: {result.E * 0.001:3f} GPa\n"
            f"\tyield stress: {result.yield_stress:3f} MPa\n"
            f"----------------------------------------\n"
            f"Limit Performance:\n"
            f"\ttensile strength: {DataProcessor.analyze_tensile_strength(s):3f} MPa\n"
            f"\tfracture strength: {DataProcessor.analyze_fracture_stress(s):3f} MPa\n"
            f"\tductility: {DataProcessor.analyze_ductility(s):5f}%\n"
            f"\ttoughness: {DataProcessor.analyze_toughness(s):6f} MPa\n"
            f"========================================")