"""Normalized (peak = 1.0) surface-temperature shape function for a
moving band heat source.

Source: `ref/Thermal Modeling Lecture Slides (rev4).pdf` slide 10,
citing Liu, Lannou, Wang & Keer (2004), "Solutions for Temperature Rise
in Stationary/Moving Bodies Caused by Surface Heating With Surface
Convection," J. Heat Transfer 126:776-785, Eq. (2.13). Independently
confirmed against the Liu paper directly. Implemented (with a boundary-
branch bug -- see below) in `ref/Subsurface_thermal.m` lines 46-58, and
correctly (three genuinely distinct branches) in all four material
sheets of `ref/ME599 Spreadsheet with RS and Ti64, 304SS and
AA6061.xlsx`, where the normalized column was confirmed to numerically
peak at exactly 1.0.

Bug note: `Subsurface_thermal.m` line 52 reads
`elseif (-1 < x_locations(i))<1`, a chained comparison that MATLAB (like
Python) does not evaluate as a range test -- `(-1 < x)` reduces to a
boolean 0/1 first, so the comparison `<1` is only ever true when
`-1 < x` is false, a case already fully handled by the preceding `if`.
Every point with `-1 < x/b < 1` therefore silently falls through to the
`x/b > 1` formula in the reference script, producing a real
discontinuity at `x/b = -1`. This implementation uses the correct
three-branch form below (the branches are constructed so that the
"left"/"middle" formulas agree in the limit at `x/b = -1`, and the
"middle"/"right" formulas agree in the limit at `x/b = +1` -- i.e. the
correct function is continuous there, unlike the buggy one).
"""

import numpy as np
from scipy.special import k0, k1

_EPS = 1e-9


def _safe_arg(z: np.ndarray) -> np.ndarray:
    """Nudge a modified-Bessel-K argument away from exactly 0 (singular
    there), preserving its sign, so grid points landing exactly on the
    x/b = -1 or x/b = +1 branch boundary don't produce inf/nan. Serves
    the same role as Subsurface_thermal.m's off-integer grid (start
    -2.001, step 0.1), which avoids the singularity by construction."""
    z = np.asarray(z, dtype=float)
    return np.where(np.abs(z) < _EPS, np.where(z >= 0, _EPS, -_EPS), z)


def _raw_shape(x_over_b, Pe: float) -> np.ndarray:
    x = np.asarray(x_over_b, dtype=float)

    P_plus = _safe_arg(Pe * (x + 1.0) / 2.0)
    P_minus = _safe_arg(Pe * (x - 1.0) / 2.0)

    left = (x + 1.0) * np.exp(P_plus) * (k0(-P_plus) - k1(-P_plus)) + (
        1.0 - x
    ) * np.exp(P_minus) * (k0(-P_minus) - k1(-P_minus))

    middle = (x + 1.0) * np.exp(P_plus) * (k0(P_plus) + k1(P_plus)) + (
        1.0 - x
    ) * np.exp(P_minus) * (k0(-P_minus) - k1(-P_minus))

    right = (x + 1.0) * np.exp(P_plus) * (k0(P_plus) + k1(P_plus)) + (
        1.0 - x
    ) * np.exp(P_minus) * (k0(P_minus) + k1(P_minus))

    return np.select([x < -1.0, x > 1.0], [left, right], default=middle)


def normalized_shape(x_over_b, Pe: float) -> np.ndarray:
    """Dimensionless surface-temperature shape, peaking at exactly 1.0,
    for the given Peclet number `Pe`, evaluated over the (array-like)
    dimensionless coordinate `x_over_b`."""
    raw = _raw_shape(x_over_b, Pe)
    return raw / np.max(raw)
