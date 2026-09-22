"""Tests for the 2D (x/b, z/b) subsurface temperature field.

Run with: pytest
"""

import numpy as np
import pytest

from cutting_model.materials import TI64
from cutting_model.model import CuttingThermalModel
from cutting_model.subsurface import subsurface_temperature


@pytest.fixture
def ti64_model():
    # Subsurface_thermal.m defaults.
    return CuttingThermalModel(material=TI64, v_m_min=60.0, Fc_N=50.0, w_mm=3.0, b_um=200.0)


def test_surface_matches_1d_model(ti64_model):
    profile = ti64_model.solve()
    field = ti64_model.solve_2d(z_over_b=np.array([0.0]))
    assert np.allclose(
        field.T_field_C[0],
        np.interp(field.x_over_b, profile.x_over_b, profile.T_actual_C),
        atol=1e-6,
    )


def test_temperature_nonincreasing_with_depth(ti64_model):
    field = ti64_model.solve_2d(z_over_b=np.linspace(0.0, 4.0, 50))
    # At the flank column, temperature should never rise as z increases.
    flank_idx = int(np.argmin(np.abs(field.x_over_b - 1.0)))
    T_col = field.T_field_C[:, flank_idx]
    assert np.all(np.diff(T_col) <= 1e-9)


def test_temperature_floors_at_ambient_far_from_surface():
    result = subsurface_temperature(
        x_over_b=np.array([0.5]),
        z_over_b=np.array([100.0]),  # far below the surface
        Pe=30.0,
        T_flash=200.0,
        T_ambient=20.0,
        k=6.6,
        cp=550.0,
        rho=4500.0,
        v_m_s=1.0,
        b_m=2e-4,
    )
    assert result[0, 0] == pytest.approx(20.0)


def test_subsurface_field_is_finite(ti64_model):
    field = ti64_model.solve_2d()
    assert np.isfinite(field.T_field_C).all()
    assert field.T_field_C.min() >= 20.0 - 1e-9
