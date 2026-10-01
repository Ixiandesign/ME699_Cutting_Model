# Cutting Thermal Model

Peclet-normalized tool-chip contact temperature model for Ti-6Al-4V, with 2D subsurface
temperature, residual stress and critical cutting speed. It reproduces
`ref/ME599 Spreadsheet with RS and Ti64, 304SS and AA6061.xlsx` (sheets `0.02 (master)` and
`Ti64`), shown in a Streamlit UI and a notebook.

```
uv sync
uv run python app.py     # launches the UI in the browser
uv run jupyter lab cutting_model.ipynb   # same results as a notebook
uv run pytest            # tests
```

## Files (all at root)
- `model.py` — all physics: force, shear-plane contact width and heat partition, flash temperature, surface shape, subsurface field, residual stress, speed sweeps and critical-speed fits
- `cutting_model.ipynb` — notebook walkthrough: inputs cell at top, equations, same outputs as the UI
- `app.py` — the whole UI
- `test_model.py` — tests (several pin values from the workbook)
- `ref/` — source papers, slides, spreadsheets

## Notes
- Validated against the workbook: all 250 speed rows of the five feed sheets (every column), the LOGEST/LINEST fits, the iteration table and the critical-speed table agree to ~1e-14. The only difference is the log-fit critical speed (~5e-6 relative): the sheet writes e as 2.71828.
- Inputs are limited to the sheet's range (v 6-300 m/min, h up to 0.12 mm); the low-Pe polynomial C4(Pe) diverges beyond it.
- The result depends on the initial temperature guess, because b, R and Pe(shear) are fixed from properties at that temperature. The default is halfway between ambient and melting (840 °C); the sheet uses hand-picked guesses of 20-210 °C per feed, which give a lower flash temperature (about 398-415 °C vs 483 °C at v = 60 m/min, h = 0.05 mm) and shift critical speeds by up to ~8%.
- Sheet conventions kept as-is: the flash temperature rise is used as the property temperature on the next speed row and compared with Tc directly; the first high-Pe pass omits the factor 2 that later passes include; the critical speed uses Tc = 500 °C (sheet input) while residual stress uses the computed thermal-yield temperature, 480 °C.
- Fixed a dead-branch bug from `ref/Subsurface_thermal.m` line 52 (chained comparison); `test_shape_continuous_at_branch_boundaries` guards it.
