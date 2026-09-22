"""Regression / correctness tests for the cutting thermal model.

Run with: pytest
"""

import numpy as np
import pytest

from cutting_model.flash_temperature import solve_flash_temperature
from cutting_model.forces import estimate_cutting_force
from cutting_model.materials import AA7050, TI64
from cutting_model.peclet import peclet_number
from cutting_model.shape_function import normalized_shape


@pytest.mark.parametrize("Pe", [0.1, 0.5, 5.0, 14.925, 50.0])
def test_shape_peaks_at_one(Pe):
    x = np.linspace(-3.0, 5.0, 2000)
    shape = normalized_shape(x, Pe)
    assert np.isfinite(shape).all()
    assert np.max(shape) == pytest.approx(1.0)


def test_peclet_number_matches_reference_workbook():
    # ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx,
    # Ti64 sheet, reference-condition block (rows 4/9): v=4 m/s,
    # rho=4500, b=2e-5 m, cp=547.32, k=6.6009 -> Pe = 14.9250.
    Pe = peclet_number(v_m_s=4.0, rho=4500.0, b_m=2e-5, cp=547.32, k=6.6009)
    assert Pe == pytest.approx(14.925, rel=1e-3)


@pytest.mark.parametrize("Pe", [0.5, 5.0, 14.925, 50.0])
def test_shape_continuous_at_branch_boundaries(Pe):
    # This is the check that would have caught Subsurface_thermal.m's
    # dead elseif branch (line 52): under that bug, the x/b > 1 formula
    # gets silently applied for -1 < x/b < 1 too, producing a real jump
    # at x/b = -1 that this test would fail on.
    #
    # The raw (pre-normalization) shape involves K0/K1 terms that blow
    # up individually near the boundary (K1(z) ~ 1/z) but cancel against
    # a vanishing prefactor -- the two branches provably converge to the
    # same limit, but only *logarithmically* slowly (verified numerically
    # during implementation: relative left/right disagreement was ~3.5%
    # at delta=1e-4, ~0.06% at delta=1e-6, ~1e-5% at delta=1e-8). So this
    # test uses a small-but-not-extreme delta and a correspondingly loose
    # tolerance, rather than delta=1e-4 (too coarse to show agreement)
    # or delta=1e-10 (too close to float64 precision limits).
    delta = 1e-6
    for boundary in (-1.0, 1.0):
        x = np.array([boundary - delta, boundary + delta])
        shape = normalized_shape(x, Pe)
        assert shape[0] == pytest.approx(shape[1], rel=1e-2)


def test_peak_shifts_toward_leading_edge_as_pe_increases():
    # Qualitative check against Liu et al. (2004) Fig. 5-7 and the
    # Peclet paper's Fig. 12: the shape function's peak moves from near
    # the center toward x/b = 1 as Pe grows.
    x = np.linspace(-3.0, 5.0, 4000)
    peak_low = x[np.argmax(normalized_shape(x, 0.5))]
    peak_high = x[np.argmax(normalized_shape(x, 50.0))]
    assert peak_high > peak_low


def test_flash_temperature_converges_for_matlab_defaults():
    # Subsurface_thermal.m defaults: Ti-6Al-4V, v=1 m/s, Fc=50 N,
    # w=3 mm, b=200 um.
    result = solve_flash_temperature(TI64, v_m_s=1.0, Fc_N=50.0, w_m=0.003, b_m=0.0002)
    assert result.converged
    assert result.T_flash > 20.0  # heated above ambient
    assert np.isfinite(result.Pe)
    # Regression baseline, pinned from this corrected implementation
    # (not hand-derivable in advance -- the fixed-point iteration has no
    # closed form). If this ever changes, it should be because a formula
    # was deliberately revised, not by accident.
    assert result.T_flash == pytest.approx(193.81, abs=0.5)
    assert result.Pe == pytest.approx(30.94, abs=0.1)


def test_estimate_cutting_force_ti64_positive():
    Fc = estimate_cutting_force(TI64, h_mm=0.1, w_mm=3.0)
    assert Fc > 0


def test_estimate_cutting_force_rejects_unsupported_material():
    with pytest.raises(ValueError):
        estimate_cutting_force(AA7050, h_mm=0.1, w_mm=3.0)
