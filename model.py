"""All physics for the Peclet-normalized cutting thermal model.

Pipeline (see solve / solve_2d at the bottom):
  1. flash_temperature: peak temperature rise T_flash and Peclet number Pe
  2. normalized_shape:  surface profile T(x/b), scaled to peak = 1
  3. subsurface_temperature: 2D field T(x/b, z/b) via erf depth attenuation
  4. critical_temperature + residual_stress: thermal-yield onset and RS vs depth

Surface:  T(x) = T_ambient + T_flash * shape(x/b)

Sources: ref/Thermal Modeling Lecture Slides (rev4).pdf, ref/Subsurface_thermal.m,
ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx,
ref/Cutting Force Data.xlsx, Liu et al. 2004 (ASME), Theraroz et al. 2025.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import erf, k0e, k1e

# ================================================================ materials
# k, cp, E, alpha, sigma_y are functions of temperature T (deg C) so one
# solver handles every material. Units: k W/(m K), cp J/(kg K), E and
# sigma_y MPa, alpha 1/degC. rs_fit = (slope, intercept): RS[MPa] =
# slope*T - intercept for T above the critical temperature.
# Values hardcoded from the reference workbook's material sheets.
# `defaults` = starting process parameters for the UI.
#
# Reference-data quirks (carried over as-is, treat as approximations):
#  - AA6061's thermal block is identical to AA7050's (likely copied).
#  - AA7050 and AA6061 reuse Ti-6Al-4V's rs_fit; AA6061 reuses AA7050's alpha.
#    So RS can come out slightly negative just above their critical temps.
MATERIALS = {
    "Ti-6Al-4V": dict(
        rho=4500.0,
        k=lambda T: 1.12e-5 * T**2 + 0.00982 * T + 6.4,
        cp=lambda T: 1.71e-6 * T**3 - 0.00173 * T**2 + 0.6 * T + 536,
        E=lambda T: -40.0 * T + 100000.0,
        alpha=lambda T: (1.23e-6 * T**2 + 2.87e-3 * T + 4.57) * 1.8e-6,
        sigma_y=lambda T: (0.005752 * T**4 - 10.67 * T**3 + 5597.0 * T**2
                           - 1.79e6 * T + 1.056e9) * 1e-6,
        poisson=0.32,
        rs_fit=(2.8788, 1365.2),
        defaults=dict(b_um=200.0, w_mm=3.0, Fc_N=50.0, v_m_min=60.0),
    ),
    "AA7050": dict(
        rho=2700.0,
        k=lambda T: 167.0,
        cp=lambda T: 896.0,
        E=lambda T: 72000.0,
        alpha=lambda T: 6.957e-9 * T + 2.346e-5,
        sigma_y=lambda T: -0.0057 * T**2 + 0.00357 * T + 523.0,
        poisson=0.33,
        rs_fit=(2.8788, 1365.2),
        defaults=dict(b_um=200.0, w_mm=5.0, Fc_N=90.0, v_m_min=240.0),
    ),
    "304 SS": dict(
        rho=8030.0,
        k=lambda T: 16.2,
        cp=lambda T: 500.0,
        E=lambda T: 193000.0,
        alpha=lambda T: 3.474e-9 * T + 1.643e-5,
        sigma_y=lambda T: (2.243e-9 * T**4 - 5.194e-6 * T**3 + 4.193e-3 * T**2
                           - 1.452 * T + 325.0),
        poisson=0.29,
        rs_fit=(5.5589, 482.0),
        defaults=dict(b_um=80.0, w_mm=5.0, Fc_N=70.0, v_m_min=18.0),
    ),
    "AA6061": dict(
        rho=2700.0,
        k=lambda T: 167.0,
        cp=lambda T: 896.0,
        E=lambda T: 70000.0,
        alpha=lambda T: 6.957e-9 * T + 2.346e-5,
        sigma_y=lambda T: (-9.41258882776867e-08 * T**4 + 1.0097053648153174e-04 * T**3
                           - 3.409959217736532e-02 * T**2 + 3.002017033412448 * T
                           + 220.85501992639655),
        poisson=0.33,
        rs_fit=(2.8788, 1365.2),
        defaults=dict(b_um=200.0, w_mm=5.0, Fc_N=90.0, v_m_min=240.0),
    ),
}

# Kienzle fit kc(h) = C * h**n (h in mm, kc in N/mm^2) from ref/Cutting Force
# Data.xlsx. Ti-6Al-4V only: the force data for other alloys doesn't match the
# alloys that have thermal data.
TI64 = "Ti-6Al-4V"
KIENZLE_TI64 = (1792.69, -0.11672)


def kienzle_force(h_mm, w_mm):
    """Tangential cutting force Fc (N) from feed h and width of cut w (Ti-6Al-4V)."""
    C, n = KIENZLE_TI64
    return C * h_mm**n * h_mm * w_mm


# ================================================================ 1. flash T
def peclet_number(v_m_s, rho, b_m, cp, k):
    """Pe = v*rho*b*cp / (2k). b = contact half-width."""
    return v_m_s * rho * b_m * cp / (2.0 * k)


@dataclass(frozen=True)
class FlashResult:
    T_flash: float
    Pe: float
    iterations: int
    converged: bool
    k: float   # conductivity at convergence, W/(m K)
    cp: float  # specific heat at convergence, J/(kg K)


def flash_temperature(material, v_m_s, Fc_N, w_m, b_m, T_ambient=20.0,
                      max_iter=200, tol=1e-3):
    """Peak temperature rise by damped fixed-point iteration (k, cp depend on T).

    Follows Subsurface_thermal.m (x2 on Fc on both Pe branches; the Excel
    workbooks omit it on the high-Pe branch, a discrepancy in the references).
    Adds an iteration cap (the script has none; extreme speeds can diverge)
    and a numeric tolerance instead of integer rounding.
    """
    m = MATERIALS[material]
    rho = m["rho"]
    T_guess = T_ambient
    converged = False
    Pe, T_flash = float("nan"), T_guess
    k, cp = m["k"](T_guess), m["cp"](T_guess)
    iteration = 0

    for iteration in range(1, max_iter + 1):
        k, cp = m["k"](T_guess), m["cp"](T_guess)
        Pe = peclet_number(v_m_s, rho, b_m, cp, k)
        if Pe < 5:
            C4 = 0.00527 * Pe**3 - 0.192 * Pe**2 + 2.39 * Pe
            T_flash = 0.159 * C4 * (2.0 * Fc_N) / (rho * cp * w_m * b_m)
        else:
            T_flash = (0.399 * (2.0 * Fc_N * v_m_s) / (k * w_m)
                       * (k / (rho * cp * v_m_s * b_m)) ** 0.5)
        if abs(T_flash - T_guess) < tol:
            converged = True
            break
        T_guess = (T_flash + T_guess) / 2.0

    return FlashResult(T_flash, Pe, iteration, converged, k, cp)


# ============================================================ 2. surface shape
def _safe(z):
    """Nudge Bessel-K arguments off exactly 0 (singular there), keeping sign."""
    return np.where(np.abs(z) < 1e-9, np.where(z >= 0, 1e-9, -1e-9), z)


def normalized_shape(x_over_b, Pe):
    """Surface temperature shape for a moving band source (Liu 2004 Eq. 2.13):
    three branches (x<-1, -1<x<1, x>1), divided by its own max so peak = 1.

    Subsurface_thermal.m line 52 has a dead middle branch (a chained
    comparison that is never true); this uses the correct three-branch form.
    """
    x = np.asarray(x_over_b, dtype=float)
    Pp = _safe(Pe * (x + 1.0) / 2.0)
    Pm = _safe(Pe * (x - 1.0) / 2.0)

    # exp(P)*K(|P|) written with scaled Bessel functions (k0e = exp(|P|)*k0)
    # so large Pe (fast cutting) doesn't overflow.
    def pos(P):  # exp(P) * (K0(P) + K1(P)),  P > 0
        P = np.abs(P)
        return k0e(P) + k1e(P)

    def neg(P):  # exp(P) * (K0(-P) - K1(-P)),  P < 0
        P = np.abs(P)
        return np.exp(-2.0 * P) * (k0e(P) - k1e(P))

    plus_neg, plus_pos = (x + 1.0) * neg(Pp), (x + 1.0) * pos(Pp)
    minus_neg, minus_pos = (1.0 - x) * neg(Pm), (1.0 - x) * pos(Pm)

    raw = np.select([x < -1.0, x > 1.0],
                    [plus_neg + minus_neg, plus_pos + minus_pos],
                    default=plus_pos + minus_neg)
    return raw / np.max(raw)


# ========================================================== 3. subsurface field
def subsurface_temperature(x_over_b, z_over_b, Pe, T_flash, T_ambient,
                           k, cp, rho, v_m_s, b_m):
    """2D field T(x/b, z/b), shape (len(z), len(x)).

    Separable approximation (ref slides 11-12, Subsurface_thermal.m, Excel):
    surface temperature times a 1D transient-conduction decay
    1 - erf(z / sqrt(4*alpha*t)), alpha = k/(rho cp), t = 2b/v. The references
    attenuate the whole temperature (ambient included) then floor at ambient;
    reproduced as-is. k, cp are the values at the converged flash temperature.
    """
    shape = normalized_shape(x_over_b, Pe)
    T_surface = T_ambient + T_flash * shape
    diffusion_length = np.sqrt(4.0 * (k / (rho * cp)) * (2.0 * b_m / v_m_s))
    atten = 1.0 - erf(np.asarray(z_over_b)[:, None] * b_m / diffusion_length)
    return np.maximum(T_ambient, atten * T_surface[None, :])


# ========================================================= 4. residual stress
@lru_cache(maxsize=32)
def critical_temperature(material, T_ambient=20.0, T_max=2000.0, n=20000):
    """Lowest T where thermoelastic stress E*alpha*(T-T0)/(1-nu) first exceeds
    yield strength sigma_y(T). Returns inf if it never does below T_max."""
    m = MATERIALS[material]
    T = np.linspace(T_ambient, T_max, n)
    sigma_thermal = m["E"](T) * m["alpha"](T) * (T - T_ambient) / (1.0 - m["poisson"])
    exceeds = sigma_thermal > m["sigma_y"](T)
    return float(T[np.argmax(exceeds)]) if exceeds.any() else float("inf")


def residual_stress(material, T, T_critical):
    """Residual stress (MPa): slope*T - intercept where T > T_critical, else 0.
    The fits are independent lines (not forced through zero at T_critical), so
    small negative values just above T_critical are a fit artifact."""
    slope, intercept = MATERIALS[material]["rs_fit"]
    T = np.asarray(T, dtype=float)
    return np.where(T > T_critical, slope * T - intercept, 0.0)


# ================================================================== one-call API
# x/b in [-3, 5] captures the peak (it moves toward x/b = 1 at high Pe) and the
# far-field decay; z/b in [0, 4] covers the subsurface gradient.
DEFAULT_X_OVER_B = np.linspace(-3.0, 5.0, 2000)
DEFAULT_Z_OVER_B = np.linspace(0.0, 4.0, 200)


@dataclass(frozen=True)
class ProfileResult:
    x_over_b: np.ndarray
    shape: np.ndarray  # normalized, peak = 1.0
    T_actual_C: np.ndarray
    Pe: float
    T_flash: float
    peak_x_over_b: float
    converged: bool
    iterations: int
    k: float
    cp: float


@dataclass(frozen=True)
class TwoDResult:
    x_over_b: np.ndarray
    z_over_b: np.ndarray
    T_field_C: np.ndarray  # (len(z), len(x))
    T_flank_profile_C: np.ndarray  # (len(z),) at the flank column
    residual_stress_MPa: np.ndarray  # (len(z),)
    T_critical_C: float
    flank_x_over_b: float
    Pe: float
    T_flash: float
    converged: bool


def _flash(material, v_m_min, Fc_N, w_mm, b_um, T_ambient):
    return flash_temperature(material, v_m_min / 60.0, Fc_N, w_mm / 1000.0,
                             b_um * 1e-6, T_ambient)


def solve(material, v_m_min, Fc_N, w_mm, b_um, T_ambient=20.0, x_over_b=None):
    """Flash temperature plus normalized and actual surface profiles."""
    x = DEFAULT_X_OVER_B if x_over_b is None else np.asarray(x_over_b, dtype=float)
    f = _flash(material, v_m_min, Fc_N, w_mm, b_um, T_ambient)
    shape = normalized_shape(x, f.Pe)
    return ProfileResult(x, shape, T_ambient + f.T_flash * shape, f.Pe, f.T_flash,
                         float(x[np.argmax(shape)]), f.converged, f.iterations,
                         f.k, f.cp)


def solve_2d(material, v_m_min, Fc_N, w_mm, b_um, T_ambient=20.0,
             x_over_b=None, z_over_b=None, flank_x_over_b=1.0):
    """Subsurface field plus residual stress vs depth at the flank/tool-exit
    column (x/b = flank_x_over_b, default 1.0)."""
    x = DEFAULT_X_OVER_B if x_over_b is None else np.asarray(x_over_b, dtype=float)
    z = DEFAULT_Z_OVER_B if z_over_b is None else np.asarray(z_over_b, dtype=float)
    f = _flash(material, v_m_min, Fc_N, w_mm, b_um, T_ambient)

    field = subsurface_temperature(x, z, f.Pe, f.T_flash, T_ambient, f.k, f.cp,
                                   MATERIALS[material]["rho"], v_m_min / 60.0,
                                   b_um * 1e-6)
    flank_idx = int(np.argmin(np.abs(x - flank_x_over_b)))
    T_flank = field[:, flank_idx]
    T_crit = critical_temperature(material, T_ambient)
    return TwoDResult(x, z, field, T_flank, residual_stress(material, T_flank, T_crit),
                      T_crit, float(x[flank_idx]), f.Pe, f.T_flash, f.converged)


# ============================================== machining sweeps (assignment)
# Force, flash temperature and critical speed vs feed h (uncut chip thickness)
# and cutting speed v (m/s). Force comes from the Kienzle fit, so these are
# Ti-6Al-4V only. Contact half-width b and width of cut w are held fixed.
_FLANK_X = np.linspace(-3.0, 5.0, 801)  # contains x/b = 1.0 exactly
_FLANK_IDX = 500


def force_vs_feed(h_mm, w_mm=3.0):
    """Estimated cutting force Fc (N) for each feed h (mm)."""
    return kienzle_force(np.asarray(h_mm, dtype=float), w_mm)


def flash_vs_speed(v_m_s, h_mm, w_mm=3.0, b_um=200.0, T_ambient=20.0):
    """Peak surface temperature (deg C) for each cutting speed (m/s) at feed h."""
    Fc = float(kienzle_force(h_mm, w_mm))
    return np.array([
        T_ambient + flash_temperature(TI64, v, Fc, w_mm / 1000.0, b_um * 1e-6,
                                      T_ambient).T_flash
        for v in np.atleast_1d(v_m_s)])


def _flank_surface_temperature(v_m_s, h_mm, w_mm, b_um, T_ambient):
    """Surface temperature at the flank/tool-exit point x/b = 1, the location
    where residual stress is evaluated."""
    Fc = float(kienzle_force(h_mm, w_mm))
    f = flash_temperature(TI64, v_m_s, Fc, w_mm / 1000.0, b_um * 1e-6, T_ambient)
    return T_ambient + f.T_flash * normalized_shape(_FLANK_X, f.Pe)[_FLANK_IDX]


def critical_speed(h_mm, w_mm=3.0, b_um=200.0, T_ambient=20.0,
                   v_min=0.01, v_max=10.0):
    """Lowest cutting speed (m/s) at which the flank surface temperature
    exceeds Ti-6Al-4V's critical temperature (thermal yielding, hence surface
    tensile residual stress). nan if never reached by v_max; v_min if already
    exceeded there. Log-spaced scan, then bisection."""
    Tc = critical_temperature(TI64, T_ambient)
    over = lambda v: _flank_surface_temperature(v, h_mm, w_mm, b_um, T_ambient) > Tc
    grid = np.geomspace(v_min, v_max, 60)
    hits = [i for i, v in enumerate(grid) if over(v)]
    if not hits:
        return float("nan")
    if hits[0] == 0:
        return float(v_min)
    lo, hi = grid[hits[0] - 1], grid[hits[0]]
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if over(mid) else (mid, hi)
    return float(hi)


def critical_speed_fit(h_mm, v_crit):
    """Power-law fit v_crit = a * h**n (h in mm, v in m/s) to the finite
    points. Returns (a, n)."""
    h, v = np.asarray(h_mm, dtype=float), np.asarray(v_crit, dtype=float)
    ok = np.isfinite(v)
    n, ln_a = np.polyfit(np.log(h[ok]), np.log(v[ok]), 1)
    return float(np.exp(ln_a)), float(n)
