"""Tests for the thermoelastic-stress-vs-yield residual-stress model.

Run with: pytest
"""

import numpy as np
import pytest

from cutting_model.materials import AA6061, AA7050, SS304, TI64
from cutting_model.model import CuttingThermalModel
from cutting_model.residual_stress import critical_temperature, residual_stress


@pytest.mark.parametrize(
    "material,expected_Tc",
    [
        (TI64, 479.58),
        (AA7050, 161.78),
        (SS304, 72.87),
        (AA6061, 117.92),
    ],
)
def test_critical_temperature_regression(material, expected_Tc):
    # Regression baseline, pinned from this implementation (not
    # hand-derivable in advance -- each material's crossing point comes
    # from a quartic/cubic sigma_y(T) vs a linear-in-T sigma_thermal(T)).
    # If this ever changes, it should be because a property fit was
    # deliberately revised, not by accident.
    Tc = critical_temperature(material)
    assert Tc == pytest.approx(expected_Tc, abs=0.05)


def test_residual_stress_zero_below_critical():
    T = np.array([20.0, 100.0, 200.0])
    RS = residual_stress(TI64, T, T_critical=479.58)
    assert np.all(RS == 0.0)


def test_residual_stress_matches_fit_above_critical():
    T_critical = 479.58
    T = np.array([500.0, 800.0])
    RS = residual_stress(TI64, T, T_critical)
    slope, intercept = TI64.rs_fit
    assert np.allclose(RS, slope * T - intercept)


def test_matlab_default_ti64_has_no_residual_stress():
    # Subsurface_thermal.m defaults (v=1 m/s, Fc=50N, w=3mm, b=200um)
    # converge to T_flash ~= 193.8 degC, well below Ti64's ~480 degC
    # critical temperature -- no plastic yielding occurs anywhere.
    model = CuttingThermalModel(material=TI64, v_m_min=60.0, Fc_N=50.0, w_mm=3.0, b_um=200.0)
    result = model.solve_2d()
    assert result.T_critical_C == pytest.approx(479.58, abs=0.05)
    assert np.all(result.residual_stress_MPa == 0.0)


def test_high_temperature_scenario_produces_residual_stress():
    # A more aggressive cutting condition pushes Ti64's flash
    # temperature well above its critical temperature, so a near-
    # surface tensile residual-stress zone should appear at the flank.
    model = CuttingThermalModel(material=TI64, v_m_min=250.0, Fc_N=400.0, w_mm=3.0, b_um=200.0)
    result = model.solve_2d()
    assert result.T_flash > result.T_critical_C
    assert result.residual_stress_MPa.max() > 0.0
    # Residual stress should appear only near the surface (shallow z),
    # since temperature decays with depth.
    nonzero = np.nonzero(result.residual_stress_MPa)[0]
    assert nonzero.max() < len(result.z_over_b) // 2
