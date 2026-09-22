from .flash_temperature import FlashResult, solve_flash_temperature
from .materials import AA6061, AA7050, MATERIALS, SS304, TI64, Material
from .model import CuttingThermalModel, ProfileResult
from .peclet import peclet_number
from .shape_function import normalized_shape

__all__ = [
    "AA6061",
    "AA7050",
    "CuttingThermalModel",
    "FlashResult",
    "MATERIALS",
    "Material",
    "ProfileResult",
    "SS304",
    "TI64",
    "normalized_shape",
    "peclet_number",
    "solve_flash_temperature",
]
