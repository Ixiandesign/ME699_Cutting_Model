"""Material property library for the cutting thermal model.

Values are hardcoded from
`ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx` (material
property blocks) and `ref/Cutting Force Data.xlsx` (Ti-6Al-4V Kienzle
fit), not read from those files at runtime.

`k` and `cp` are stored as callables of temperature (deg C) so that the
same iterative flash-temperature solver works uniformly for every
material, even the ones with constant (temperature-independent)
properties.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Material:
    name: str
    rho: float  # kg/m^3
    k: Callable[[float], float]  # W/(m*K), function of temperature (deg C)
    cp: Callable[[float], float]  # J/(kg*K), function of temperature (deg C)


def _ti64_k(T: float) -> float:
    return 1.12e-5 * T**2 + 0.00982 * T + 6.4


def _ti64_cp(T: float) -> float:
    return 1.71e-6 * T**3 - 0.00173 * T**2 + 0.6 * T + 536


TI64 = Material(name="Ti-6Al-4V", rho=4500.0, k=_ti64_k, cp=_ti64_cp)

AA7050 = Material(name="AA7050", rho=2700.0, k=lambda T: 167.0, cp=lambda T: 896.0)

SS304 = Material(name="304 SS", rho=8030.0, k=lambda T: 16.2, cp=lambda T: 500.0)

# AA6061: the reference workbook's thermal-property block for AA6061
# ("AA60601" sheet) was numerically identical, to every digit, to
# AA7050's block -- almost certainly cloned and never updated with
# AA6061-specific values, rather than an independent measurement. Carried
# over here with the same flag; treat as an approximation.
AA6061 = Material(name="AA6061", rho=2700.0, k=lambda T: 167.0, cp=lambda T: 896.0)

MATERIALS = {m.name: m for m in (TI64, AA7050, SS304, AA6061)}

# Default process parameters per material (contact half-width, width of
# cut, cutting force, cutting speed), taken from the reference workbook's
# demo rows / Subsurface_thermal.m's defaults (Ti-6Al-4V).
DEFAULTS = {
    TI64.name: dict(b_um=200.0, w_mm=3.0, Fc_N=50.0, v_m_min=60.0),
    AA7050.name: dict(b_um=200.0, w_mm=5.0, Fc_N=90.0, v_m_min=240.0),
    SS304.name: dict(b_um=80.0, w_mm=5.0, Fc_N=70.0, v_m_min=18.0),
    AA6061.name: dict(b_um=200.0, w_mm=5.0, Fc_N=90.0, v_m_min=240.0),
}
