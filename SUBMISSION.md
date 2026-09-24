# Initial thermal calculator submission

Start with `uv run streamlit run app.py` and open http://localhost:8501.
Maximize your browser for readable side-by-side plots. A narrow window stacks
the overview vertically, so scroll to capture each numbered section.

## Screenshots

1. Click **A · Ti64 baseline (50 N)** in the sidebar. Capture **Assignment
   overview**, including inputs, flash-temperature metrics, and numbered plots.
2. Click **B · Ti64 tensile stress (200 N)**. Capture the same view to demonstrate
   the change in temperature and the nonzero subsurface tensile stress.
3. Use **1D view** and **2D view** for larger graphs. The 2D view includes both
   the temperature field and the residual-stress-versus-depth graph.
4. Copy the narrative below or capture the narrative at the bottom of the overview.

Both examples use Ti-6Al-4V, speed 60 m/min, contact half-width b = 200 µm,
width of cut w = 3 mm, ambient temperature 20 °C, and directly entered force.
The full contact length is 2b = 400 µm. No feed-based force calculation is needed.

| Output | A: 50 N | B: 200 N |
|---|---:|---:|
| Flash temperature rise | 193.8 K | 566.0 K |
| Peak surface temperature | 213.8 °C | 586.0 °C |
| Peclet number | 30.936 | 18.278 |
| Thermal conductivity at final iterate | 8.72 W/(m·K) | 15.55 W/(m·K) |
| Specific heat at final iterate | 599.75 J/(kg·K) | 631.44 J/(kg·K) |
| Thermal-yield threshold | 479.6 °C | 479.6 °C |
| Temperature at selected subsurface point | 192.8 °C | 532.1 °C |
| Tensile residual stress at that point | 0.0 MPa | 166.6 MPa |

The selected subsurface point is the UI grid's x/b = 0.987 (approximately 1),
z/b = 0.02685, or z = 5.37 µm. Density is 4500 kg/m³. These are illustrative
model predictions, not experimental validation. The zero result in A is
meaningful: it remains below the thermal-yield threshold.

## Brief narrative

This calculator takes cutting force, speed, contact half-width, width of cut,
and material properties as inputs. It iterates a Peclet-based flash-temperature
correlation, then scales a normalized moving-band-source surface profile by
the temperature rise and adds the 20 °C ambient temperature. A transient
conduction depth attenuation, using exposure time 2b/v, produces an approximate
2D subsurface temperature field. Near x/b = 1, comparison of constrained
thermoelastic stress with temperature-dependent yield strength establishes a
critical temperature; above this threshold, the Ti-6Al-4V course-reference fit
σres = 2.8788T − 1365.2 MPa estimates tensile residual stress, with zero assigned
below the threshold. The model is limited by its separable depth approximation,
fixed effective diffusivity, empirical stress correlation, and omission of
mechanically induced residual stress. It also inherits reference conventions
that evaluate thermal properties at the flash-rise iterate and attenuate total
Celsius temperature before clipping at ambient; these introduce physical
approximations, and extrapolating the property fits may be unreliable.

## Requirement coverage

- **1:** Flash rise and absolute peak temperature are separate numerical outputs;
  force, geometry, speed, density, conductivity, and specific heat are visible.
- **2:** Surface temperature versus x/b appears in the overview and 1D tab.
- **3:** Temperature versus x/b and z/b appears as a labeled heatmap with a °C
  color scale in the overview and 2D tab.
- **4:** Ti-6Al-4V stress versus depth appears near x/b = 1, with an explicit
  numerical subsurface tensile-stress estimate in the overview.

The implementation retains the equations in `ref/Subsurface_thermal.m`, with
the existing corrected surface branch logic. The reference equations and
limitations are documented in `cutting_model/` and `README.md`.
