# Peclet-Normalized Cutting Temperature & Residual Stress Model

A Python model of the tool-chip/flank contact temperature in orthogonal
metal cutting, built around a Peclet number, extended to a 2D subsurface
temperature field and a residual-stress estimate. The core surface
deliverable is a **dimensionless shape curve, normalized to a peak of
exactly 1.0**, as a function of position `x/b` along the contact
(`b` = contact half-width), parameterized by the Peclet number `Pe`.
That curve is then rescaled by a separately-computed peak ("flash")
temperature to get the actual temperature-rise curve in degrees C:

```
T(x) = T_ambient + T_flash * shape(x/b)
```

That surface profile is then extended into the subsurface (depth `z/b`)
by a 1D transient-conduction erf attenuation, and the resulting
thermal cycle at the flank (`x/b = 1`) is compared against each
material's temperature-dependent yield strength to estimate a
residual-stress-vs-depth profile.

An interactive matplotlib desktop app (`app.py`) lets you vary material,
cutting speed, force, and contact geometry and toggle between a **1D
view** (surface shape + actual temperature) and a **2D view**
(subsurface temperature field + residual stress at the flank), all
updating live.

## Quick start

Dependencies and the virtualenv are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync              # creates .venv, installs deps + uv.lock
uv run python app.py # interactive UI
uv run pytest        # test suite
```

## Physics summary

All formulas are cross-validated across four sources in `ref/`: the
course lecture slides, a 2025 journal paper (Theraroz, Tuysuz & Schoop),
the Liu et al. (2004) ASME paper the slides cite, and a reference MATLAB
script / Excel workbooks from the same lineage of course work. Full
derivation notes, citations, and documented discrepancies between
sources are in `cutting_model/*.py` docstrings and in the implementation
plans this project was built from
(`C:\Users\kaden\.claude\plans\this-repo-is-for-rosy-frog.md` for the
original 1D surface model, `this-is-a-thermo-zany-rabbit.md` for the
2D subsurface + residual-stress extension).

1. **Peclet number**: `Pe = v * rho * b * cp / (2 * k)`
2. **Flash (peak) temperature**: a two-branch (`Pe` above/below 5)
   closed-form correlation, solved by fixed-point iteration since `k`
   and `cp` depend on temperature for some materials.
3. **Normalized shape function**: a three-branch closed form built from
   modified Bessel functions `K0`/`K1` (the moving band heat source
   solution), divided by its own peak so it always equals 1.0 at its
   maximum, wherever that maximum falls for the given `Pe`.

## Subsurface temperature & residual stress

The subsurface (depth) temperature field `T(x/b, z/b)` attenuates the
surface shape with a 1D semi-infinite-solid erf solution (`ref/Thermal
Modeling Lecture Slides (rev4).pdf` slides 11-12, citing Ozisik Ch. 9),
evaluated at the exposure time `t = 2*b/v_c` a material point spends
under the moving contact. This is a *separable approximation*, not an
exact solution of the 2D moving-source PDE -- `ref/transient heat
transfer moving souce Liu 2004 ASME.pdf` gives an exact point-source
Green's function valid at any depth, but states no closed form exists
for a finite band source under motion, and neither reference source
(the MATLAB script or the Excel workbook) uses it, so this
implementation doesn't either. See `cutting_model/subsurface.py` for
details, including a note on the formula's non-smooth decay toward
ambient (both reference sources do it identically, so it's reproduced
as-is).

Residual stress (`cutting_model/residual_stress.py`) compares a
biaxially-constrained thermoelastic stress
`sigma_thermal(T) = E(T)*alpha_cte(T)*(T-T_ambient)/(1-nu)` against
each material's temperature-dependent yield strength `sigma_y(T)` to
find a "critical temperature" `T_critical` (the lowest T at which
thermal stress first exceeds yield). Points at the flank (`x/b = 1`)
whose local peak temperature exceeds `T_critical` get a residual
stress from each material's pre-fit linear correlation
`RS(T) = slope*T - intercept` (an inverse-calibrated fit against
XRD-measured data, per `ref/Peclet paper 2025.pdf` Eq. 38); points that
never exceed `T_critical` get `RS = 0`.

All four materials' thermoelastic properties (`E`, `alpha_cte`,
`sigma_y`, `poisson`, `rs_fit`) were read directly from
`ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx`'s
per-material sheets (verified against the literal cell formulas, not
just displayed values). Two caveats carried over from that workbook:

- **AA7050 and AA6061's `rs_fit` are numerically identical to
  Ti-6Al-4V's** -- almost certainly copy-pasted rather than
  independently regressed. Only 304 stainless steel has its own fit.
  A concrete downstream effect: since AA7050/AA6061's own critical
  temperatures are much lower than Ti64's, `RS(T)` comes out
  *negative* over a wide band of temperatures just above their
  critical temperature (before turning positive again near 474°C,
  where Ti64's fit happens to cross zero) -- an artifact of reusing
  the wrong material's fit, not a physically meaningful compressive
  result. 304SS has the same issue but over a much narrower band.
- **AA6061's `alpha_cte` is identical to AA7050's** -- the same
  duplication already flagged for their thermal `k`/`cp`.

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
    subsurface.py             2D (x/b, z/b) subsurface temperature field
    residual_stress.py        thermoelastic-yield residual-stress model
    model.py                  ties the above together for the UI
app.py                        interactive matplotlib UI (1D / 2D view toggle)
tests/                        pytest suite (formula checks, bug regression, convergence)
```
