"""Run with: pytest"""

import numpy as np
import pytest

from model import (critical_speed, critical_speed_fit, critical_temperature,
                   flash_temperature, flash_vs_speed, force_vs_feed, kienzle_force,
                   normalized_shape, peclet_number, residual_stress, solve,
                   solve_2d, subsurface_temperature)


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
    f = flash_temperature(1.0, 50.0, 0.003, 0.0002, a_low=1.0, a_high=1.0)
    assert f.converged
    assert f.T_flash == pytest.approx(193.81, abs=0.5)
    assert f.Pe == pytest.approx(30.94, abs=0.1)


def test_kienzle_force_positive():
    assert kienzle_force(0.1, 3.0) > 0


# ------------------------------------------------------------- subsurface
def test_surface_matches_1d_model():
    args = (60.0, 50.0, 3.0, 200.0)
    profile = solve(*args)
    field = solve_2d(*args, z_over_b=np.array([0.0]))
    assert np.allclose(
        field.T_field_C[0],
        np.interp(field.x_over_b, profile.x_over_b, profile.T_actual_C),
        atol=1e-6,
    )


def test_temperature_nonincreasing_with_depth():
    field = solve_2d(60.0, 50.0, 3.0, 200.0, z_over_b=np.linspace(0.0, 4.0, 50))
    flank_idx = int(np.argmin(np.abs(field.x_over_b - 1.0)))
    assert np.all(np.diff(field.T_field_C[:, flank_idx]) <= 1e-9)


def test_temperature_floors_at_ambient_far_from_surface():
    result = subsurface_temperature(
        x_over_b=np.array([0.5]), z_over_b=np.array([100.0]), Pe=30.0,
        T_flash=200.0, T_ambient=20.0, k=6.6, cp=550.0, rho=4500.0,
        v_m_s=1.0, b_m=2e-4)
    assert result[0, 0] == pytest.approx(20.0)


def test_subsurface_field_is_finite():
    field = solve_2d(60.0, 50.0, 3.0, 200.0)
    assert np.isfinite(field.T_field_C).all()
    assert field.T_field_C.min() >= 20.0 - 1e-9


# ---------------------------------------------------------- residual stress
def test_critical_temperature_regression():
    assert critical_temperature() == pytest.approx(479.58, abs=0.05)


def test_residual_stress_zero_below_critical():
    RS = residual_stress(np.array([20.0, 100.0, 200.0]), T_critical=479.58)
    assert np.all(RS == 0.0)


def test_residual_stress_matches_fit_above_critical():
    T = np.array([500.0, 800.0])
    RS = residual_stress(T, 479.58)
    assert np.allclose(RS, 2.8788 * T - 1365.2)


def test_matlab_default_ti64_has_no_residual_stress():
    # T_flash ~194 C is well below Ti64's ~480 C critical temperature.
    result = solve_2d(60.0, 50.0, 3.0, 200.0)
    assert result.T_critical_C == pytest.approx(479.58, abs=0.05)
    assert np.all(result.residual_stress_MPa == 0.0)


def test_high_temperature_scenario_produces_residual_stress():
    result = solve_2d(250.0, 400.0, 3.0, 200.0)
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


# ------------------------------------------------------ machining sweeps
def test_force_increases_with_feed():
    F = force_vs_feed(np.linspace(0.01, 0.10, 10))
    assert np.all(np.diff(F) > 0)


def test_flash_increases_with_speed_and_feed():
    v = np.array([0.1, 1.0, 10.0])
    lo, hi = flash_vs_speed(v, 0.01), flash_vs_speed(v, 0.10)
    assert np.all(np.isfinite(lo)) and np.all(np.diff(lo) > 0)
    assert np.all(hi > lo)


def test_critical_speed_decreases_with_feed():
    vc = [critical_speed(h) for h in (0.08, 0.10, 0.12)]
    assert np.all(np.diff(vc) < 0)
    a, n = critical_speed_fit([0.08, 0.10, 0.12], vc)
    assert n < 0 and a > 0


def test_high_pe_shape_does_not_overflow():
    assert np.isfinite(normalized_shape(np.linspace(-3, 5, 100), 300.0)).all()


# ------------------------------------------- reference workbook (Ti64 sheet)
def test_matches_ti64_example_sheet():
    """Single pass at 20 C properties from 'ME599 Spreadsheet with RS ...xlsx',
    sheet Ti64: v = 4 m/s, Fc = 50 N, b = 20 um, w = 5 mm."""
    f = flash_temperature(4.0, 50.0, 0.005, 2e-5, T_initial=20.0, a_low=1.0, a_high=1.0)
    assert f.history[1] == pytest.approx(885.0939818816958, rel=1e-9)
    from model import TI64
    k, cp = TI64["k"](20.0), TI64["cp"](20.0)
    Pe = peclet_number(4.0, 4500.0, 2e-5, cp, k)
    x = -2.001 + 0.1 * np.arange(44)  # sheet's x/b grid; index 30 is x/b = 0.999
    field = subsurface_temperature(x, np.array([0.0, 0.1, 0.2]), Pe,
                                   885.0939818816958, 20.0, k, cp, 4500.0, 4.0, 2e-5)
    # Sheet row 62 (x/b = 0.999), z/b = 0, 0.1, 0.2, with Tc = 480
    assert residual_stress(field[:, 30], 480.0) == pytest.approx(
        [1079.627715711081, 553.3011790874748, 64.59577961665127], rel=1e-6)


def test_initial_guess_is_midpoint_of_ambient_and_melt():
    assert flash_temperature(1.0, 50.0, 0.003, 2e-4).history[0] == pytest.approx(840.0)


def test_calibration_fits_data_and_is_continuous():
    from model import CALIBRATION, DATA_T_C, DATA_V_M_MIN, DATA_H, calibrate_flash
    cal = calibrate_flash()
    assert cal["a_high"] == pytest.approx(CALIBRATION["a_high"], abs=1e-3)
    assert cal["a_low"] == pytest.approx(CALIBRATION["a_low"], abs=1e-3)
    assert cal["rms"] < 40.0
    T = flash_vs_speed(np.array(DATA_V_M_MIN) / 60.0, DATA_H)
    assert np.all(np.abs(T - np.array(DATA_T_C)) < 70.0)
