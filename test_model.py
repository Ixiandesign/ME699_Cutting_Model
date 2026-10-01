"""Run with: pytest"""

import numpy as np
import pytest

from model import (critical_temperature, flash_temperature, kienzle_force,
                   normalized_shape, peclet_number, residual_stress, solve,
                   solve_2d, subsurface_temperature)

TI64 = "Ti-6Al-4V"


# ---------------------------------------------------------------- 1D shape
@pytest.mark.parametrize("Pe", [0.1, 0.5, 5.0, 14.925, 50.0])
def test_shape_peaks_at_one(Pe):
    shape = normalized_shape(np.linspace(-3.0, 5.0, 2000), Pe)
    assert np.isfinite(shape).all()
    assert np.max(shape) == pytest.approx(1.0)


def test_peclet_matches_reference_workbook():
    Pe = peclet_number(v_m_s=4.0, rho=4500.0, b_m=2e-5, cp=547.32, k=6.6009)
    assert Pe == pytest.approx(14.925, rel=1e-3)


@pytest.mark.parametrize("Pe", [0.5, 5.0, 14.925, 50.0])
def test_shape_continuous_at_branch_boundaries(Pe):
    # Would catch the dead-middle-branch bug in Subsurface_thermal.m line 52.
    # Left/right limits converge only logarithmically, hence delta and rel.
    delta = 1e-6
    for boundary in (-1.0, 1.0):
        shape = normalized_shape(np.array([boundary - delta, boundary + delta]), Pe)
        assert shape[0] == pytest.approx(shape[1], rel=1e-2)


def test_peak_shifts_toward_leading_edge_as_pe_increases():
    x = np.linspace(-3.0, 5.0, 4000)
    assert x[np.argmax(normalized_shape(x, 50.0))] > x[np.argmax(normalized_shape(x, 0.5))]


# --------------------------------------------------------- flash + force
def test_flash_temperature_matlab_defaults():
    f = flash_temperature(TI64, 1.0, 50.0, 0.003, 0.0002)
    assert f.converged
    assert f.T_flash == pytest.approx(193.81, abs=0.5)
    assert f.Pe == pytest.approx(30.94, abs=0.1)


def test_kienzle_force_positive():
    assert kienzle_force(0.1, 3.0) > 0


# ------------------------------------------------------------- subsurface
def test_surface_matches_1d_model():
    args = (TI64, 60.0, 50.0, 3.0, 200.0)
    profile = solve(*args)
    field = solve_2d(*args, z_over_b=np.array([0.0]))
    assert np.allclose(
        field.T_field_C[0],
        np.interp(field.x_over_b, profile.x_over_b, profile.T_actual_C),
        atol=1e-6,
    )


def test_temperature_nonincreasing_with_depth():
    field = solve_2d(TI64, 60.0, 50.0, 3.0, 200.0, z_over_b=np.linspace(0.0, 4.0, 50))
    flank_idx = int(np.argmin(np.abs(field.x_over_b - 1.0)))
    assert np.all(np.diff(field.T_field_C[:, flank_idx]) <= 1e-9)


def test_temperature_floors_at_ambient_far_from_surface():
    result = subsurface_temperature(
        x_over_b=np.array([0.5]), z_over_b=np.array([100.0]), Pe=30.0,
        T_flash=200.0, T_ambient=20.0, k=6.6, cp=550.0, rho=4500.0,
        v_m_s=1.0, b_m=2e-4)
    assert result[0, 0] == pytest.approx(20.0)


def test_subsurface_field_is_finite():
    field = solve_2d(TI64, 60.0, 50.0, 3.0, 200.0)
    assert np.isfinite(field.T_field_C).all()
    assert field.T_field_C.min() >= 20.0 - 1e-9


# ---------------------------------------------------------- residual stress
@pytest.mark.parametrize("material,expected_Tc", [
    (TI64, 479.58), ("AA7050", 161.78), ("304 SS", 72.87), ("AA6061", 117.92)])
def test_critical_temperature_regression(material, expected_Tc):
    assert critical_temperature(material) == pytest.approx(expected_Tc, abs=0.05)


def test_residual_stress_zero_below_critical():
    RS = residual_stress(TI64, np.array([20.0, 100.0, 200.0]), T_critical=479.58)
    assert np.all(RS == 0.0)


def test_residual_stress_matches_fit_above_critical():
    T = np.array([500.0, 800.0])
    RS = residual_stress(TI64, T, 479.58)
    assert np.allclose(RS, 2.8788 * T - 1365.2)


def test_matlab_default_ti64_has_no_residual_stress():
    # T_flash ~194 C is well below Ti64's ~480 C critical temperature.
    result = solve_2d(TI64, 60.0, 50.0, 3.0, 200.0)
    assert result.T_critical_C == pytest.approx(479.58, abs=0.05)
    assert np.all(result.residual_stress_MPa == 0.0)


def test_high_temperature_scenario_produces_residual_stress():
    result = solve_2d(TI64, 250.0, 400.0, 3.0, 200.0)
    assert result.T_flash > result.T_critical_C
    assert result.residual_stress_MPa.max() > 0.0
    nonzero = np.nonzero(result.residual_stress_MPa)[0]
    assert nonzero.max() < len(result.z_over_b) // 2  # only near the surface


# ---------------------------------------------------------------------- app
def test_direct_launch_enters_streamlit_cli():
    """`python app.py` must hand off to Streamlit rather than run in bare mode."""
    import subprocess
    import sys
    from pathlib import Path

    app = Path(__file__).resolve().parent / "app.py"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(app), "--help"], cwd=app.parent,
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "missing ScriptRunContext" not in output
    assert "--server.port" in output
