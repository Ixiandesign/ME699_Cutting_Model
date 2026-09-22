"""Cutting-force-from-feed estimation via a Kienzle-type power-law fit.

Only Ti-6Al-4V has a matched force-vs-feed fit alongside its thermal
properties in the provided reference data (`ref/Cutting Force Data.xlsx`);
the aluminum/steel entries in that force workbook are different specific
alloys (7075 Al, 1045 steel, 410SS) than the ones with thermal properties
in `ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx`
(AA7050, 304SS, AA6061), so they don't line up. For those three
materials, cutting force must be entered directly.
"""

from .materials import Material, TI64

# kc(h) = C * h**n, h in mm, kc in N/mm^2 (specific cutting force).
# Fit from ref/Cutting Force Data.xlsx, Ti-6Al4V column (LOGEST power-law
# regression on the raw force-vs-feed table), R^2 = 0.9732.
KIENZLE_TI64_KC = (1792.69, -0.11672)


def estimate_cutting_force(material: Material, h_mm: float, w_mm: float) -> float:
    """Estimate tangential cutting force Fc (N) from feed/chip thickness.

    Fc = kc(h) * h * w, with kc(h) = C * h**n (Kienzle-type specific
    cutting force fit). Raises ValueError for materials without a
    matched fit in the reference data (see module docstring).
    """
    if material is not TI64:
        raise ValueError(
            f"No cutting-force-vs-feed fit is available for {material.name!r} "
            "in the provided reference data; enter the cutting force directly."
        )
    C, n = KIENZLE_TI64_KC
    kc = C * h_mm**n
    return kc * h_mm * w_mm
