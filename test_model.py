"""Run with: pytest

Reference numbers are cached values from
ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx
(sheets '0.02 (master)' and 'Ti64').
"""

import numpy as np
import pytest

from model import (TI64, SHEET_FEEDS, Constants, critical_temperature, cutting_force,
                   critical_speed_fit, feed_sweep, flash_row, initial_guess,
                   normalized_shape, peclet_number, residual_stress, solve,
                   speed_sweep, subsurface_temperature)

SHEET_GUESS = {0.015: 20.0, 0.03: 100.0, 0.05: 180.0, 0.08: 180.0, 0.12: 210.0}


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
    # Left/right limits converge only logarithmically, hence delta and rel.
    delta = 1e-6
    for boundary in (-1.0, 1.0):
        shape = normalized_shape(np.array([boundary - delta, boundary + delta]), Pe)
        assert shape[0] == pytest.approx(shape[1], rel=1e-2)


def test_peak_shifts_toward_leading_edge_as_pe_increases():
    x = np.linspace(-3.0, 5.0, 4000)
    assert x[np.argmax(normalized_shape(x, 50.0))] > x[np.argmax(normalized_shape(x, 0.5))]


def test_high_pe_shape_does_not_overflow():
    assert np.isfinite(normalized_shape(np.linspace(-3, 5, 100), 300.0)).all()


# ------------------------------------------- master sheet: one speed row
def test_cutting_force_matches_sheet():
    assert cutting_force(0.03, 3.0) == pytest.approx(170.25412860768057, rel=1e-12)
    assert cutting_force(0.05, 3.0) == pytest.approx(267.2945465051952, rel=1e-12)


def test_first_row_matches_sheet():
    """Sheet '0.02 (master)' row 21: h = 0.03, v = 0.1 m/s, Ti = 100 C."""
    r = flash_row(0.1, 100.0, 0.03, 3.0)
    assert r.b == pytest.approx(9.611447188795385e-05, rel=1e-10)
    assert r.partition == pytest.approx(0.821617689354759, rel=1e-10)
    assert r.Pe_first == pytest.approx(1.6749139500146306, rel=1e-10)
    assert r.low == pytest.approx((195.78721148872694, 170.36048967452587, 176.75226745997,
                                   175.12213759201245, 175.53637234828025), rel=1e-10)
    assert r.high == pytest.approx((122.07778680014044, 238.58226855524833, 213.94324300826878,
                                    218.61049635113548, 217.7078545262429), rel=1e-10)
    assert r.Pe == pytest.approx(1.4964147619594401, rel=1e-10)
    assert r.T_flash == pytest.approx(174.44281676869713, rel=1e-10)  # Pe < 5: low chain


def test_high_pe_row_uses_high_chain():
    """Sheet row 60 (v = 4 m/s) has Pe > 5, so BB = BA."""
    rows = speed_sweep(0.03, 3.0, 100.0)
    assert rows[39].v_m_s == pytest.approx(4.0)
    assert rows[39].Pe > 5.0
    assert rows[39].T_flash == pytest.approx(495.8712123182618, rel=1e-10)
    assert rows[49].T_flash == pytest.approx(512.9068066680532, rel=1e-10)


def test_speed_sweep_carries_temperature_between_rows():
    rows = speed_sweep(0.03, 3.0, 100.0)
    assert rows[0].T_initial == 100.0
    assert all(b.T_initial == a.T_flash for a, b in zip(rows, rows[1:]))


def test_initial_guess_is_midpoint_of_ambient_and_melt():
    assert initial_guess() == pytest.approx(840.0)


# --------------------------------------- master sheet: fits and critical speed
def test_fits_and_critical_speed_match_sheet():
    fs = feed_sweep(0.03, 3.0, 100.0)
    assert fs.power_fit == pytest.approx((141.63204941990813, 0.23293964429094918), rel=1e-9)
    assert fs.log_fit == pytest.approx((84.09268611399054, 37.16953188037132), rel=1e-9)
    assert fs.v_crit_power == pytest.approx(224.7597158350457, rel=1e-9)
    # sheet uses 2.71828 for e in the log-fit speed, hence the looser tolerance
    assert fs.v_crit_log == pytest.approx(245.6259576616407, rel=1e-4)
    assert fs.v_crit_avg == pytest.approx(235.1928367483432, rel=1e-4)


def test_critical_speed_decreases_with_feed():
    vc = [feed_sweep(h, 3.0, SHEET_GUESS[h]).v_crit_avg for h in SHEET_FEEDS]
    assert np.all(np.diff(vc) < 0)
    a, n = critical_speed_fit(SHEET_FEEDS, vc)
    assert a > 0 and n < 0


def test_flash_increases_with_feed():
    T = [flash_row(2.0, 300.0, h, 3.0).T_flash for h in (0.02, 0.05, 0.12)]
    assert np.all(np.diff(T) > 0)


def test_force_increases_with_feed():
    assert np.all(np.diff(cutting_force(np.linspace(0.01, 0.12, 12), 3.0)) > 0)


# ----------------------------------------------- 'Ti64' sheet: subsurface + RS
def test_critical_temperature_matches_sheet():
    assert critical_temperature() == 480.0  # sheet I61


def test_residual_stress_zero_below_critical():
    RS = residual_stress(np.array([20.0, 100.0, 200.0]), T_critical=480.0)
    assert np.all(RS == 0.0)


def test_residual_stress_matches_fit_above_critical():
    T = np.array([500.0, 800.0])
    assert np.allclose(residual_stress(T, 480.0), 2.8788 * T - 1365.2)


def test_subsurface_matches_ti64_sheet():
    """Sheet 'Ti64': v = 4 m/s, b = 20 um, Tf = 885.09 K, properties at 20 C;
    row 62 is RS at the flank column x/b = 0.999, z/b = 0, 0.1, 0.2."""
    k, cp = TI64["k"](20.0), TI64["cp"](20.0)
    Pe = peclet_number(4.0, 4500.0, 2e-5, cp, k)
    x = -2.001 + 0.1 * np.arange(44)  # sheet's x/b grid; index 30 is x/b = 0.999
    field = subsurface_temperature(x, np.array([0.0, 0.1, 0.2]), Pe,
                                   885.0939818816958, k, cp, 4.0, 2e-5)
    assert residual_stress(field[:, 30], 480.0) == pytest.approx(
        [1079.627715711081, 553.3011790874748, 64.59577961665127], rel=1e-6)


# ------------------------------------------------------------- full solution
def test_surface_row_matches_1d_profile():
    s = solve(120.0, 0.05, 3.0)
    assert np.allclose(s.T_field_C[0], s.T_surface_C, atol=1e-6)


def test_temperature_nonincreasing_with_depth_at_flank():
    s = solve(120.0, 0.05, 3.0)
    assert np.all(np.diff(s.T_flank_profile_C) <= 1e-9)


def test_temperature_floors_at_ambient():
    s = solve(120.0, 0.05, 3.0)
    assert np.isfinite(s.T_field_C).all()
    assert s.T_field_C.min() >= 20.0 - 1e-9


def test_flank_column_is_sheet_value():
    assert solve(120.0, 0.05, 3.0).flank_x_over_b == pytest.approx(0.999)


def test_tensile_residual_stress_only_near_surface():
    s = solve(120.0, 0.12, 3.0)
    assert s.row.T_flash > s.T_critical_C
    nonzero = np.nonzero(s.residual_stress_MPa)[0]
    assert s.residual_stress_MPa[0] > 0 and nonzero.max() < len(s.z_over_b) // 4


def test_low_speed_has_no_residual_stress():
    assert np.all(solve(20.0, 0.02, 3.0).residual_stress_MPa == 0.0)


def test_constants_change_result():
    base = solve(120.0, 0.05, 3.0).row.T_flash
    assert solve(120.0, 0.05, 3.0, Constants(a_low=1.0, a_high=1.0)).row.T_flash > base


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
