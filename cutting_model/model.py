"""High-level orchestration: bundles a material and process parameters,
and solves the flash temperature plus the normalized/actual temperature
profile in one call, for use by the UI layer."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .flash_temperature import solve_flash_temperature
from .materials import Material
from .shape_function import normalized_shape

# x/b in [-3, 5]: wide/fine enough to capture the peak across the Pe
# range of interest (it shifts toward x/b = 1 at high Pe, per Liu et al.
# Fig. 5-7 and the Peclet paper's Fig. 12) and to show the far-field
# decay on both sides.
DEFAULT_X_OVER_B = np.linspace(-3.0, 5.0, 2000)


@dataclass(frozen=True)
class ProfileResult:
    x_over_b: np.ndarray
    shape: np.ndarray  # normalized, peak = 1.0
    T_actual_C: np.ndarray
    Pe: float
    T_flash: float
    peak_x_over_b: float
    converged: bool


@dataclass
class CuttingThermalModel:
    material: Material
    v_m_min: float  # cutting speed, m/min (UI-friendly unit)
    Fc_N: float  # cutting force, N
    w_mm: float  # width of cut, mm
    b_um: float  # tool-chip/flank contact half-width, micrometers
    T_ambient_C: float = 20.0

    def solve(self, x_over_b: Optional[np.ndarray] = None) -> ProfileResult:
        if x_over_b is None:
            x_over_b = DEFAULT_X_OVER_B
        x_over_b = np.asarray(x_over_b, dtype=float)

        v_m_s = self.v_m_min / 60.0
        w_m = self.w_mm / 1000.0
        b_m = self.b_um * 1e-6

        flash = solve_flash_temperature(
            self.material, v_m_s, self.Fc_N, w_m, b_m, T_ambient=self.T_ambient_C
        )
        shape = normalized_shape(x_over_b, flash.Pe)
        T_actual = self.T_ambient_C + flash.T_flash * shape
        peak_x = float(x_over_b[np.argmax(shape)])

        return ProfileResult(
            x_over_b=x_over_b,
            shape=shape,
            T_actual_C=T_actual,
            Pe=flash.Pe,
            T_flash=flash.T_flash,
            peak_x_over_b=peak_x,
            converged=flash.converged,
        )
