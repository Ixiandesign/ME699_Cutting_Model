"""Peclet number for the moving tool-chip/flank contact heat source.

    Pe = v * rho * b * cp / (2 * k)

Source: `ref/Thermal Modeling Lecture Slides (rev4).pdf` slide 8;
`ref/Peclet paper 2025.pdf` Fig. 12 inset; `ref/Subsurface_thermal.m`
line 17. All three sources agree on this exact form.
"""


def peclet_number(v_m_s: float, rho: float, b_m: float, cp: float, k: float) -> float:
    """Peclet number.

    v_m_s: cutting speed (m/s)
    rho: density (kg/m^3)
    b_m: tool-chip/flank contact half-width (m)
    cp: specific heat (J/(kg*K))
    k: thermal conductivity (W/(m*K))
    """
    return (v_m_s * rho * b_m * cp) / (2.0 * k)
