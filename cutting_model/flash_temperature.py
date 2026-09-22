"""Peak ("flash") temperature at the tool-chip/flank contact.

Source: `ref/Thermal Modeling Lecture Slides (rev4).pdf` slide 9;
`ref/Subsurface_thermal.m` lines 13-29 (fixed-point iteration, since
k(T) and cp(T) depend on the very temperature being solved for).

The "x2" on Fc is applied on *both* branches here, matching
Subsurface_thermal.m exactly. Note: the Excel OLD APPROACH / NEW
APPROACH workbooks only apply that factor on the low-Pe branch (the
high-Pe branch there omits it) -- a discrepancy in the reference
material itself. This implementation follows the MATLAB script and the
lecture slide's own formula rather than the Excel variant.
"""

from dataclasses import dataclass

from .materials import Material
from .peclet import peclet_number


@dataclass(frozen=True)
class FlashResult:
    T_flash: float
    Pe: float
    iterations: int
    converged: bool


def _shape_coefficient(Pe: float) -> float:
    """C4(Pe), the low-Pe correction polynomial used in the flash-
    temperature formula (slide 9; Subsurface_thermal.m line 18)."""
    return 0.00527 * Pe**3 - 0.192 * Pe**2 + 2.39 * Pe


def solve_flash_temperature(
    material: Material,
    v_m_s: float,
    Fc_N: float,
    w_m: float,
    b_m: float,
    T_ambient: float = 20.0,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> FlashResult:
    """Damped fixed-point solve for the flash temperature.

    Mirrors Subsurface_thermal.m's `while true` loop (each new guess is
    the average of the previous guess and the newly computed flash
    temperature) but with a max-iteration safety cap -- the reference
    script has none, and one of the Excel workbooks documents a real
    divergence case at extreme cutting speed, where the temperature-
    dependent property polynomials get extrapolated far outside their
    fitted range and the iteration blows up instead of converging.

    Also uses a numeric tolerance on successive guesses rather than the
    reference script's `round()`-to-the-nearest-integer-degree equality
    check, which is a minor precision improvement, not a behavior change.
    """
    T_guess = T_ambient
    converged = False
    Pe = float("nan")
    T_flash = T_guess
    iteration = 0

    for iteration in range(1, max_iter + 1):
        k = material.k(T_guess)
        cp = material.cp(T_guess)
        rho = material.rho

        Pe = peclet_number(v_m_s, rho, b_m, cp, k)
        C4 = _shape_coefficient(Pe)

        if Pe < 5:
            T_flash = 0.159 * C4 * (2.0 * Fc_N) / (rho * cp * w_m * b_m)
        else:
            T_flash = (
                0.399
                * (2.0 * Fc_N * v_m_s)
                / (k * w_m)
                * (k / (rho * cp * v_m_s * b_m)) ** 0.5
            )

        if abs(T_flash - T_guess) < tol:
            converged = True
            break

        T_guess = (T_flash + T_guess) / 2.0

    return FlashResult(T_flash=T_flash, Pe=Pe, iterations=iteration, converged=converged)
