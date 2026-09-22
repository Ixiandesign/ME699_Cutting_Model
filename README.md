# 1D Peclet-Normalized Cutting Temperature Model

A Python model of the tool-chip/flank contact temperature in orthogonal
metal cutting, built around a Peclet number. The core deliverable is a
**dimensionless shape curve, normalized to a peak of exactly 1.0**, as a
function of position `x/b` along the contact (`b` = contact half-width),
parameterized by the Peclet number `Pe`. That curve is then rescaled by
a separately-computed peak ("flash") temperature to get the actual
temperature-rise curve in degrees C:

```
T(x) = T_ambient + T_flash * shape(x/b)
```

An interactive matplotlib desktop app (`app.py`) lets you vary material,
cutting speed, force, and contact geometry and see both curves update
live.

## Quick start

```bash
pip install -r requirements.txt
python app.py       # interactive UI
pytest               # test suite
```

## Physics summary

All formulas are cross-validated across four sources in `ref/`: the
course lecture slides, a 2025 journal paper (Theraroz, Tuysuz & Schoop),
the Liu et al. (2004) ASME paper the slides cite, and a reference MATLAB
script / Excel workbooks from the same lineage of course work. Full
derivation notes, citations, and documented discrepancies between
sources are in `cutting_model/*.py` docstrings and in
`C:\Users\kaden\.claude\plans\this-repo-is-for-rosy-frog.md` (the
implementation plan this project was built from).

1. **Peclet number**: `Pe = v * rho * b * cp / (2 * k)`
2. **Flash (peak) temperature**: a two-branch (`Pe` above/below 5)
   closed-form correlation, solved by fixed-point iteration since `k`
   and `cp` depend on temperature for some materials.
3. **Normalized shape function**: a three-branch closed form built from
   modified Bessel functions `K0`/`K1` (the moving band heat source
   solution), divided by its own peak so it always equals 1.0 at its
   maximum, wherever that maximum falls for the given `Pe`.

### A real bug found (and fixed, not replicated) in the reference material

`ref/Subsurface_thermal.m` line 52 has a MATLAB chained comparison
(`elseif (-1 < x_locations(i))<1`) that is silently never true, so the
script's middle branch is dead code and the `x/b > 1` formula gets
wrongly applied across the entire `-1 < x/b < 1` region. This is fixed
in `cutting_model/shape_function.py`; see that file's docstring and
`tests/test_thermal.py::test_shape_continuous_at_branch_boundaries`,
which is specifically designed to catch a regression of this bug.

## Materials

| Material | Notes |
|---|---|
| Ti-6Al-4V | Full temperature-dependent `k(T)`, `cp(T)`; matched cutting-force-vs-feed (Kienzle) fit available |
| AA7050 | Constant `k`, `cp`; cutting force is a direct input (no matched force-vs-feed data in `ref/`) |
| 304 SS | Constant `k`, `cp`; direct force input |
| AA6061 | Constant `k`, `cp` (flagged in code: the source spreadsheet's AA6061 property block was numerically identical to AA7050's, i.e. likely copied rather than independently measured); direct force input |

Only Ti-6Al-4V has both thermal properties and force-vs-feed data that
actually match in the provided reference material (the force workbook's
other entries are different specific alloys than the ones with thermal
properties). In the app, check "Estimate Fc from feed h (Kienzle fit)"
to drive cutting force from a feed-rate slider for Ti-6Al-4V; for the
other three materials, cutting force is always a direct slider input.

## Project layout

```
cutting_model/
    materials.py           material property + default-parameter library
    forces.py               Kienzle force-from-feed fit (Ti-6Al-4V)
    peclet.py                Peclet number
    flash_temperature.py     peak/flash temperature (fixed-point solve)
    shape_function.py        normalized (peak=1) surface shape function
    model.py                  ties the above together for the UI
app.py                        interactive matplotlib UI
tests/test_thermal.py         pytest suite (formula checks, bug regression, convergence)
```

## Scope

This model covers the tool-chip/flank **surface contact profile**
(`T` vs `x/b`) only. The reference material in `ref/` also contains a
subsurface/depth (`z/b`) profile and a downstream residual-stress
calculation -- both deliberately out of scope here, but structured so
they could be added as additional modules following the same pattern
(see the implementation plan for details).
