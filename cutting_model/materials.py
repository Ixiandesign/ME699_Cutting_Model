"""Material property library for the cutting thermal model.

Values are hardcoded from
`ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx` (material
property blocks) and `ref/Cutting Force Data.xlsx` (Ti-6Al-4V Kienzle
fit), not read from those files at runtime.

`k` and `cp` are stored as callables of temperature (deg C) so that the
same iterative flash-temperature solver works uniformly for every
material, even the ones with constant (temperature-independent)
properties. `E`, `alpha_cte`, and `sigma_y` (also callables of
temperature, deg C) are the residual-stress properties, read directly
from each material's sheet in the same workbook (rows 61-1042: `E` in
column C, `alpha_cte` in column D, `sigma_y` in column E -- all
verified against the literal cell formulas via `openpyxl`, not just
values). `poisson` is that sheet's fixed `F62`. `rs_fit` is the
(slope, intercept) pair from the sheet's final residual-stress formula
(`L62`), `RS(T) = slope*T - intercept` for `T` above the material's
critical temperature -- see `cutting_model/residual_stress.py`.
"""

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class Material:
    name: str
    rho: float  # kg/m^3
    k: Callable[[float], float]  # W/(m*K), function of temperature (deg C)
    cp: Callable[[float], float]  # J/(kg*K), function of temperature (deg C)
    E: Callable[[float], float]  # elastic modulus, MPa, function of temperature (deg C)
    alpha_cte: Callable[[float], float]  # linear thermal expansion coeff, 1/degC
    sigma_y: Callable[[float], float]  # yield strength, MPa, function of temperature (deg C)
    poisson: float
    rs_fit: Tuple[float, float]  # (slope, intercept): RS[MPa] = slope*T[degC] - intercept


def _ti64_k(T: float) -> float:
    return 1.12e-5 * T**2 + 0.00982 * T + 6.4


def _ti64_cp(T: float) -> float:
    return 1.71e-6 * T**3 - 0.00173 * T**2 + 0.6 * T + 536


def _ti64_E(T: float) -> float:
    return -40.0 * T + 100000.0


def _ti64_alpha(T: float) -> float:
    return (1.23e-6 * T**2 + 2.87e-3 * T + 4.57) * 1.8e-6


def _ti64_sigma_y(T: float) -> float:
    return (
        0.005752 * T**4 - 10.67 * T**3 + 5597.0 * T**2 - 1.79e6 * T + 1.056e9
    ) * 1e-6


TI64 = Material(
    name="Ti-6Al-4V",
    rho=4500.0,
    k=_ti64_k,
    cp=_ti64_cp,
    E=_ti64_E,
    alpha_cte=_ti64_alpha,
    sigma_y=_ti64_sigma_y,
    poisson=0.32,
    rs_fit=(2.8788, 1365.2),
)


def _aa7050_alpha(T: float) -> float:
    return 6.957e-9 * T + 2.346e-5


def _aa7050_sigma_y(T: float) -> float:
    return -0.0057 * T**2 + 0.00357 * T + 523.0


# AA7050's final residual-stress fit (2.8788, 1365.2) is numerically
# identical to Ti-6Al-4V's -- almost certainly a copy-paste from the
# Ti64 sheet rather than an independent regression against AA7050's own
# thermoelastic-stress curve, same pattern as the thermal-property
# duplication flagged below for AA6061. Carried over as-is (it's the
# only value the reference material provides), flagged for awareness.
AA7050 = Material(
    name="AA7050",
    rho=2700.0,
    k=lambda T: 167.0,
    cp=lambda T: 896.0,
    E=lambda T: 72000.0,
    alpha_cte=_aa7050_alpha,
    sigma_y=_aa7050_sigma_y,
    poisson=0.33,
    rs_fit=(2.8788, 1365.2),
)


def _ss304_alpha(T: float) -> float:
    return 3.474e-9 * T + 1.643e-5


def _ss304_sigma_y(T: float) -> float:
    return (
        2.243e-9 * T**4 - 5.194e-6 * T**3 + 4.193e-3 * T**2 - 1.452 * T + 325.0
    )


SS304 = Material(
    name="304 SS",
    rho=8030.0,
    k=lambda T: 16.2,
    cp=lambda T: 500.0,
    E=lambda T: 193000.0,
    alpha_cte=_ss304_alpha,
    sigma_y=_ss304_sigma_y,
    poisson=0.29,
    rs_fit=(5.5589, 482.0),  # 304SS's own independently-fit RS correlation
)


def _aa6061_sigma_y(T: float) -> float:
    # LINEST quartic fit from the "AA60601" sheet (cells J80:N80),
    # verified by direct cell read (not just the sheet's displayed
    # values) since a naive re-transcription of this fit is easy to get
    # wrong -- it's a 5-coefficient array formula, not a simple typed
    # formula like the other materials' fits.
    return (
        -9.41258882776867e-08 * T**4
        + 1.0097053648153174e-04 * T**3
        - 3.409959217736532e-02 * T**2
        + 3.002017033412448 * T
        + 220.85501992639655
    )


# AA6061: the reference workbook's thermal-property block for AA6061
# ("AA60601" sheet) was numerically identical, to every digit, to
# AA7050's block -- almost certainly cloned and never updated with
# AA6061-specific values, rather than an independent measurement. Carried
# over here with the same flag; treat as an approximation. Its
# alpha_cte and rs_fit below have the same issue: alpha_cte is
# identical to AA7050's, and rs_fit is identical to Ti64's.
AA6061 = Material(
    name="AA6061",
    rho=2700.0,
    k=lambda T: 167.0,
    cp=lambda T: 896.0,
    E=lambda T: 70000.0,
    alpha_cte=_aa7050_alpha,
    sigma_y=_aa6061_sigma_y,
    poisson=0.33,
    rs_fit=(2.8788, 1365.2),
)

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
