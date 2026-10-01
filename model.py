"""Peclet-normalized cutting thermal model for Ti-6Al-4V, following the
reference workbook ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx.

Pipeline (see solve / feed_sweep at the bottom):
  1. flash_row:          cutting force, contact half-width b, heat partition R and
                         flash temperature (sheet '0.02 (master)', one speed row)
  2. normalized_shape:   surface profile T(x/b), scaled to peak = 1 (sheet 'Ti64')
  3. subsurface_temperature: 2D field T(x/b, z/b) via erf depth attenuation
  4. critical_temperature + residual_stress: thermal-yield onset and RS vs depth
  5. speed_sweep / feed_sweep: T vs speed per feed, power-law and log fits,
                         critical speed (sheet '0.02 (master)', rows 21-70)

Surface:  T(x) = T_AMBIENT + T_flash * shape(x/b)

Sheet conventions reproduced as-is (see also the comments on each function):
  - the flash temperature RISE is carried to the next speed row as the
    property temperature, and compared with Tc directly (no ambient added)
  - b, R and Pe(shear) come from properties at the row's initial temperature
    and are held fixed through the five chain iterations
  - the first high-Pe pass omits the factor 2 that later passes include
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import erf, k0e, k1e

T_AMBIENT = 20.0  # deg C

# ================================================================ material
# Ti-6Al-4V properties as functions of temperature T (deg C). Units: k W/(m K),
# cp J/(kg K), E and sigma_y MPa, alpha 1/degC. rs_fit = (slope, intercept):
# RS[MPa] = slope*T - intercept for T above the critical temperature.
TI64 = dict(
    rho=4500.0,
    k=lambda T: 1.12e-5 * T**2 + 0.00982 * T + 6.4,
    cp=lambda T: 1.71e-6 * T**3 - 0.00173 * T**2 + 0.6 * T + 536,
    E=lambda T: -40.0 * T + 100000.0,
    alpha=lambda T: (1.23e-6 * T**2 + 2.87e-3 * T + 4.57) * 1.8e-6,
    sigma_y=lambda T: (0.005752 * T**4 - 10.67 * T**3 + 5597.0 * T**2
                       - 1.79e6 * T + 1.056e9) * 1e-6,
    poisson=0.32,
    rs_fit=(2.8788, 1365.2),
    T_melt=1660.0,  # deg C, liquidus
)
RHO = TI64["rho"]


@dataclass(frozen=True)
class Constants:
    """Fixed inputs of the sheet's 'Constants' block."""
    shear_angle_deg: float = 35.0
    rake_angle_deg: float = 0.0
    clearance_angle_deg: float = 5.0
    edge_radius_um: float = 10.0
    a_low: float = 0.95   # calibration factor, Pe < 5
    a_high: float = 0.90  # calibration factor, Pe > 5


# Sheet values for the machining sweeps ('0.02 (master)' G6:G10, DATA W5:X9).
SHEET_FEEDS = (0.015, 0.03, 0.05, 0.08, 0.12)  # mm
SHEET_DATA_H = 0.05  # mm
SHEET_DATA = ((20.0, 360.0), (40.0, 400.0), (60.0, 430.0), (80.0, 470.0), (100.0, 490.0))  # (m/min, C)
SWEEP_SPEEDS_M_S = 0.1 * np.arange(1, 51)  # 0.1 .. 5.0 m/s, as rows 21-70
T_CRIT_SPEED = 500.0  # deg C; sheet input I2, used for the critical speed


def initial_guess():
    """Starting property temperature: halfway between ambient and melting."""
    return (T_AMBIENT + TI64["T_melt"]) / 2.0


# ================================================================== force
def cutting_force(h_mm, w_mm, rake_deg=0.0):
    """Cutting force Fc (N): 1793 h^-0.117 w h ((100 - rake)/100) 0.7,
    h and w in mm (sheet cell C12)."""
    h = np.asarray(h_mm, dtype=float)
    return 1793.0 * h**-0.117 * w_mm * h * ((100.0 - rake_deg) / 100.0) * 0.7


# ============================================================ 1. flash row
def peclet_number(v_m_s, rho, b_m, cp, k):
    """Pe = v*rho*b*cp / (2k). b = contact half-width."""
    return v_m_s * rho * b_m * cp / (2.0 * k)


def _c4(Pe):
    return 0.00527 * Pe**3 - 0.192 * Pe**2 + 2.39 * Pe


@dataclass(frozen=True)
class FlashRow:
    """One speed row of the sheet (columns A..BB)."""
    v_m_s: float
    T_initial: float
    Fc: float          # N
    k: float           # W/(m K), at T_initial
    cp: float          # J/(kg K), at T_initial
    Pe_shear: float
    thermal_layer: float  # m
    b: float           # contact half-width, m
    partition: float   # R, fraction of heat into the workpiece
    Pe_first: float    # Pe(work) at T_initial
    low: tuple         # Pe<5 chain, five passes (M, S, AB, AK, AT)
    high: tuple        # Pe>5 chain, five passes (N, W, AF, AO, AX)
    Pe: float          # average Pe(work) over passes 2-5 of both chains (AY)
    T_low: float       # mean of low[1:]  (AZ)
    T_high: float      # mean of high[1:] (BA)
    T_flash: float     # BA if Pe > 5 else AZ  (BB)


def flash_row(v_m_s, T_initial, h_mm, w_mm, c=Constants()):
    """Flash temperature for cutting speed v (m/s) and feed h (mm).

    b = (l_int + l_ext)/2 from the shear-plane thermal layer and the flank
    contact length; R = min(1, 0.6 Pe_shear^-0.4); then two five-pass chains
    (low-Pe and high-Pe correlations) re-evaluating k, cp at each pass's
    temperature. T_flash is the high chain mean if the mean Pe > 5, else the
    low chain mean.
    """
    phi = np.radians(c.shear_angle_deg)
    w = w_mm * 1e-3
    ls = h_mm * 1e-3 / np.sin(phi)
    l_ext = c.edge_radius_um / (2.0 * np.tan(np.radians(c.clearance_angle_deg))) * 1e-6
    Fc = float(cutting_force(h_mm, w_mm, c.rake_angle_deg))

    k, cp = TI64["k"](T_initial), TI64["cp"](T_initial)
    Pe_shear = v_m_s * ls * RHO * cp / (4.0 * k)
    delta = np.sqrt(ls**2 / Pe_shear)
    b = (delta / np.sin(phi) + l_ext) / 2.0
    R = min(1.0, 0.6 * Pe_shear**-0.4)
    Pe_first = peclet_number(v_m_s, RHO, b, cp, k)

    def low_T(Pe, cp, k):
        return c.a_low * _c4(Pe) * 0.159 * Fc * R * 2.0 / (RHO * cp * w * b)

    def high_T(cp, k, factor):
        return (c.a_high * 0.399 * Fc * R * v_m_s * factor / (k * w)
                * np.sqrt(k / (RHO * cp * v_m_s * b)))

    low = [low_T(Pe_first, cp, k)]
    high = [high_T(cp, k, 1.0)]  # first pass has no factor 2 in the sheet (col N)
    pes = []
    for _ in range(4):
        for chain, is_low in ((low, True), (high, False)):
            kk, cc = TI64["k"](chain[-1]), TI64["cp"](chain[-1])
            Pe = peclet_number(v_m_s, RHO, b, cc, kk)
            pes.append(Pe)
            chain.append(low_T(Pe, cc, kk) if is_low else high_T(cc, kk, 2.0))

    Pe_avg = float(np.mean(pes))
    T_low, T_high = float(np.mean(low[1:])), float(np.mean(high[1:]))
    return FlashRow(v_m_s, T_initial, Fc, k, cp, Pe_shear, float(delta), float(b), R,
                    Pe_first, tuple(low), tuple(high), Pe_avg, T_low, T_high,
                    T_high if Pe_avg > 5.0 else T_low)


# ============================================================ 2. surface shape
def _safe(z):
    """Nudge Bessel-K arguments off exactly 0 (singular there), keeping sign."""
    return np.where(np.abs(z) < 1e-9, np.where(z >= 0, 1e-9, -1e-9), z)


def normalized_shape(x_over_b, Pe):
    """Surface temperature shape for a moving band source (Liu 2004 Eq. 2.13):
    three branches (x<-1, -1<x<1, x>1), divided by its own max so peak = 1."""
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
def subsurface_temperature(x_over_b, z_over_b, Pe, T_flash, k, cp, v_m_s, b_m):
    """2D field T(x/b, z/b), shape (len(z), len(x)).

    Surface temperature times a 1D transient-conduction decay
    1 - erf(z / sqrt(4*alpha*t)), alpha = k/(rho cp), t = 2b/v, floored at
    ambient. The whole temperature (ambient included) is attenuated, as in
    the sheet. k, cp are the values at the row's initial temperature.
    """
    shape = normalized_shape(x_over_b, Pe)
    T_surface = T_AMBIENT + T_flash * shape
    diffusion_length = np.sqrt(4.0 * (k / (RHO * cp)) * (2.0 * b_m / v_m_s))
    atten = 1.0 - erf(np.asarray(z_over_b)[:, None] * b_m / diffusion_length)
    return np.maximum(T_AMBIENT, atten * T_surface[None, :])


# ========================================================= 4. residual stress
def thermoelastic_stress(T):
    """Constrained thermal stress E*alpha*(T - T0)/(1 - nu), MPa."""
    return (TI64["E"](T) * TI64["alpha"](T) * (T - T_AMBIENT)
            / (1.0 - TI64["poisson"]))


@lru_cache(maxsize=None)
def critical_temperature():
    """Lowest T (1 deg C steps from ambient, as the sheet) where thermoelastic
    stress first exceeds the yield strength sigma_y(T)."""
    T = np.arange(T_AMBIENT, 1001.0)
    exceeds = thermoelastic_stress(T) > TI64["sigma_y"](T)
    return float(T[np.argmax(exceeds)]) if exceeds.any() else float("inf")


def residual_stress(T, T_critical):
    """Residual stress (MPa): slope*T - intercept where T > T_critical, else 0.
    The fit is an independent line (not forced through zero at T_critical)."""
    slope, intercept = TI64["rs_fit"]
    T = np.asarray(T, dtype=float)
    return np.where(T > T_critical, slope * T - intercept, 0.0)


# ================================================================== one-call API
# x/b grid contains 0.999 (the sheet's flank column; the -3.001 offset keeps
# samples off the x/b = +-1 singularities). z/b in [0, 5] is finer near the
# surface, where the residual stress lives.
DEFAULT_X_OVER_B = np.round(-3.001 + 0.01 * np.arange(801), 3)
DEFAULT_Z_OVER_B = 5.0 * np.linspace(0.0, 1.0, 200) ** 2
FLANK_X_OVER_B = 0.999


@dataclass(frozen=True)
class Solution:
    row: FlashRow
    x_over_b: np.ndarray
    shape: np.ndarray  # normalized, peak = 1.0
    T_surface_C: np.ndarray
    peak_x_over_b: float
    z_over_b: np.ndarray
    T_field_C: np.ndarray  # (len(z), len(x))
    flank_x_over_b: float
    T_flank_profile_C: np.ndarray  # (len(z),)
    residual_stress_MPa: np.ndarray  # (len(z),)
    T_critical_C: float

    @property
    def peak_surface_C(self):
        return T_AMBIENT + self.row.T_flash


def solve(v_m_min, h_mm, w_mm, c=Constants(), T_initial=None):
    """Flash temperature, surface profile, subsurface field and flank residual
    stress for cutting speed v (m/min) and feed h (mm)."""
    T_initial = initial_guess() if T_initial is None else T_initial
    row = flash_row(v_m_min / 60.0, T_initial, h_mm, w_mm, c)
    x, z = DEFAULT_X_OVER_B, DEFAULT_Z_OVER_B
    shape = normalized_shape(x, row.Pe)
    field = subsurface_temperature(x, z, row.Pe, row.T_flash, row.k, row.cp,
                                   row.v_m_s, row.b)
    flank_idx = int(np.argmin(np.abs(x - FLANK_X_OVER_B)))
    T_flank = field[:, flank_idx]
    T_crit = critical_temperature()
    return Solution(row, x, shape, T_AMBIENT + row.T_flash * shape,
                    float(x[np.argmax(shape)]), z, field, float(x[flank_idx]),
                    T_flank, residual_stress(T_flank, T_crit), T_crit)


# ============================================== machining sweeps (assignment)
def speed_sweep(h_mm, w_mm, T_initial, c=Constants(), v_m_s=SWEEP_SPEEDS_M_S):
    """Flash rows over increasing speed; each row starts from the previous
    row's flash temperature (the first from T_initial), as the sheet does."""
    rows, T = [], T_initial
    for v in v_m_s:
        rows.append(flash_row(float(v), T, h_mm, w_mm, c))
        T = rows[-1].T_flash
    return rows


def power_law_fit(v_m_min, T):
    """T = a * v**n by log-log regression (Excel LOGEST on ln v). Returns (a, n)."""
    n, ln_a = np.polyfit(np.log(v_m_min), np.log(T), 1)
    return float(np.exp(ln_a)), float(n)


def log_fit(v_m_min, T):
    """T = slope * ln(v) + intercept (Excel LINEST on ln v). Returns (slope, intercept)."""
    slope, intercept = np.polyfit(np.log(v_m_min), T, 1)
    return float(slope), float(intercept)


@dataclass(frozen=True)
class FeedSweep:
    h_mm: float
    rows: list
    v_m_min: np.ndarray
    T_flash: np.ndarray
    Pe: np.ndarray
    power_fit: tuple    # (a, n)
    log_fit: tuple      # (slope, intercept)
    v_crit_power: float  # m/min
    v_crit_log: float    # m/min
    v_crit_avg: float    # m/min, mean of the two (sheet column R)


def feed_sweep(h_mm, w_mm, T_initial, c=Constants(), T_crit=T_CRIT_SPEED):
    """Speed sweep at feed h plus the critical speed where the fitted flash
    temperature reaches T_crit: power law (Tc/a)^(1/n) and log fit
    exp((Tc - intercept)/slope), and their average."""
    rows = speed_sweep(h_mm, w_mm, T_initial, c)
    v = np.array([r.v_m_s for r in rows]) * 60.0
    T = np.array([r.T_flash for r in rows])
    a, n = power_law_fit(v, T)
    slope, intercept = log_fit(v, T)
    v_pl = (T_crit / a) ** (1.0 / n)
    v_log = float(np.exp((T_crit - intercept) / slope))
    return FeedSweep(h_mm, rows, v, T, np.array([r.Pe for r in rows]), (a, n),
                     (slope, intercept), float(v_pl), v_log, (float(v_pl) + v_log) / 2.0)


def critical_speed_fit(h_mm, v_crit):
    """Power-law fit v_crit = a * h**n (h in mm) to critical speed vs feed.
    Returns (a, n)."""
    n, ln_a = np.polyfit(np.log(np.asarray(h_mm, dtype=float)),
                         np.log(np.asarray(v_crit, dtype=float)), 1)
    return float(np.exp(ln_a)), float(n)
