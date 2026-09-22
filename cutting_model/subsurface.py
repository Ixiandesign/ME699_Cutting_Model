"""2D (x/b, z/b) subsurface temperature field, via depth-wise erf
attenuation of the surface shape function.

Source: `ref/Thermal Modeling Lecture Slides (rev4).pdf` slides 11-12
(1D semi-infinite-solid transient-conduction solution, citing Ozisik
Ch. 9): `T(z,t) = (1 - erf(z / sqrt(4*alpha*t))) * T_flash + T_bulk`,
`alpha = k/(rho*cp)`, `t = 2*b/v_c` (the time a material point spends
under the moving contact). Implemented identically in
`ref/Subsurface_thermal.m` lines 59-86 and, over a full (x/b, z/b)
grid rather than a single row, in every material sheet of
`ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx`
(e.g. cell M14: `=IF((1-ERF(...))*$L14<20, 20, (1-ERF(...))*$L14)`).

This is a *separable* approximation, not an exact solution of the 2D
moving-source PDE: it multiplies the already-solved surface shape by
an independent 1D transient-conduction decay factor, rather than
solving for x and z jointly. `ref/transient heat transfer moving
souce Liu 2004 ASME.pdf` Sec. 2.2.3.2 gives an exact point-source
Green's function valid at any depth
(`G(x,z) = (1/pi) * exp(Pe*x/2) * K0(Pe*R/2)`, `R = sqrt(x^2+z^2)`),
but states plainly that no closed form exists for a finite-width band
source under motion -- only numerical integration. Neither reference
source (MATLAB script or Excel workbook) uses that route; both use
the simpler erf-attenuation approximation below, so that's what's
implemented here.

Also note: the formula attenuates the *entire* surface temperature
(ambient offset included), not just the temperature rise above
ambient, then floors the result at `T_ambient`. That means it does not
decay smoothly toward `T_ambient` as `z` grows -- it decays toward 0
and gets clipped. Both reference sources do this identically, so it's
reproduced as-is rather than "corrected" (unlike the genuine
shape-function branch bug already fixed in `shape_function.py`, where
the two reference sources disagreed and one was provably wrong).
"""

import numpy as np
from scipy.special import erf

from .shape_function import normalized_shape


def subsurface_temperature(
    x_over_b,
    z_over_b,
    Pe: float,
    T_flash: float,
    T_ambient: float,
    k: float,
    cp: float,
    rho: float,
    v_m_s: float,
    b_m: float,
) -> np.ndarray:
    """2D temperature field T(x/b, z/b), shape (len(z_over_b), len(x_over_b)).

    k, cp: material properties at the converged flash temperature (see
    `FlashResult.k`/`.cp`), reused as-is for the whole field, matching
    the reference sources (which also use a single fixed diffusivity
    throughout the subsurface calculation, not a depth/temperature-
    varying one).
    """
    x_over_b = np.asarray(x_over_b, dtype=float)
    z_over_b = np.asarray(z_over_b, dtype=float)

    shape = normalized_shape(x_over_b, Pe)
    T_surface = T_ambient + T_flash * shape  # (Nx,)

    alpha = k / (rho * cp)
    t_exposure = 2.0 * b_m / v_m_s
    diffusion_length = np.sqrt(4.0 * alpha * t_exposure)

    z_phys = z_over_b[:, None] * b_m  # (Nz, 1)
    atten = 1.0 - erf(z_phys / diffusion_length)  # (Nz, 1)

    T_field = atten * T_surface[None, :]  # (Nz, Nx)
    return np.maximum(T_ambient, T_field)
