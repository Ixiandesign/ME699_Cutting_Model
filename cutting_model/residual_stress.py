"""Residual stress from a thermoelastic-stress-vs-yield-strength
comparison over the machining thermal cycle.

Source: `ref/Thermal Modeling Lecture Slides (rev4).pdf` slides 13-15
(biaxially-constrained thermoelastic stress
`sigma_thermal(T) = E(T)*alpha_cte(T)*(T-T_ambient)/(1-nu)`, compared
against temperature-dependent yield strength `sigma_y(T)`);
`ref/Peclet paper 2025.pdf` Eq. (38) (Theraroz, Tuysuz & Schoop 2025),
same functional form, described there as an *inverse-calibrated*
correlation fit against XRD-measured residual stress rather than a
first-principles derivation; `ref/Subsurface_thermal.m` lines 92-131;
`ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx` rows
61-1042 (every material sheet).

Method: as a material point is heated from ambient up through its
local peak temperature, a thermoelastic stress `sigma_thermal(T)`
develops. Once it exceeds the (temperature-dependent, generally
falling) yield strength `sigma_y(T)`, the material yields plastically;
subsequent cooling locks in tensile residual stress. The "critical
temperature" `T_critical` is the lowest T at which `sigma_thermal(T)`
first exceeds `sigma_y(T)`. Points whose local peak temperature
exceeds `T_critical` are assigned a residual stress via each
material's pre-fit linear correlation `RS(T) = slope*T - intercept`
(`Material.rs_fit`); points that never exceed `T_critical` get
`RS = 0`. This mirrors the reference workbook's `G`/`H`/`I61`/`L62`
columns exactly (see `cutting_model/materials.py` for the per-material
`E`, `alpha_cte`, `sigma_y`, `poisson`, `rs_fit` values, verified
directly against the workbook's cell formulas).

Quirk inherited from the reference `rs_fit` correlations (not corrected
here, since it's a property of the reference material's own numbers,
not a bug in this implementation): each `rs_fit = (slope, intercept)`
is an independently-fit line through XRD-measured data (per the Peclet
paper's inverse-calibration method) with no constraint that it pass
through zero at that material's own `T_critical`. For Ti-6Al-4V,
`T_critical` (~480 degC) happens to exceed the fit's own zero-crossing
(~474 degC), so `RS(T)` is positive immediately above `T_critical`. But
AA7050 and AA6061 both reuse Ti64's `rs_fit` constants (see
`materials.py`) with their own, much lower, `T_critical` (~162 degC and
~118 degC respectively) -- so for a wide band of temperatures just
above their `T_critical`, `RS(T) = slope*T - intercept` comes out
*negative* before crossing back to positive above ~474 degC. 304
stainless steel's own independently-fit correlation has the same
issue, just over a much narrower band (`T_critical` ~73 degC vs. a
zero-crossing at ~87 degC). This is left as-is, matching the reference
workbook's formula exactly; treat small negative "residual stress"
values just above a material's critical temperature as an artifact of
the fit, not a physically meaningful compressive result.
"""

from functools import lru_cache

import numpy as np

from .materials import Material


@lru_cache(maxsize=32)
def critical_temperature(
    material: Material, T_ambient: float = 20.0, T_max: float = 2000.0, n: int = 20000
) -> float:
    """Lowest temperature at which thermoelastic stress first exceeds
    yield strength, scanning upward from `T_ambient`. Returns `inf` if
    no such crossing occurs within `[T_ambient, T_max]` (the material
    never yields thermally over that range).

    Cached: this depends only on `material` (one of a handful of
    module-level singleton `Material` instances) and the scan
    parameters, never on process conditions (speed, force, geometry),
    so re-running the 20000-point scan on every UI redraw is wasted
    work."""
    T = np.linspace(T_ambient, T_max, n)
    sigma_thermal = (
        material.E(T) * material.alpha_cte(T) * (T - T_ambient) / (1.0 - material.poisson)
    )
    sigma_y = material.sigma_y(T)
    exceeds = sigma_thermal > sigma_y
    if not np.any(exceeds):
        return float("inf")
    return float(T[np.argmax(exceeds)])


def residual_stress(material: Material, T, T_critical: float) -> np.ndarray:
    """Residual stress (MPa) at each temperature in `T` (deg C), using
    `material.rs_fit = (slope, intercept)`: `slope*T - intercept` where
    `T > T_critical`, else 0."""
    T = np.asarray(T, dtype=float)
    slope, intercept = material.rs_fit
    return np.where(T > T_critical, slope * T - intercept, 0.0)
