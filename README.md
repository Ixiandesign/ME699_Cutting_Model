# Cutting Thermal Model

Peclet-normalized tool-chip contact temperature model for Ti-6Al-4V, with 2D subsurface
temperature and residual stress, shown in a Streamlit UI.

```
uv sync
uv run python app.py     # launches the UI in the browser
uv run jupyter lab cutting_model.ipynb   # same results as a notebook
uv run pytest            # tests
```

## Files (all at root)
- `model.py` — all physics: Ti-6Al-4V properties, Kienzle force fit, Peclet number, flash temperature, surface shape, subsurface field, residual stress, `solve()` / `solve_2d()`
- `cutting_model.ipynb` — notebook walkthrough: inputs cell at top, equations, same outputs as the UI
- `app.py` — the whole UI
- `test_model.py` — tests
- `ref/` — source papers, slides, spreadsheets

## Notes
- Fixed a dead-branch bug from `ref/Subsurface_thermal.m` line 52 (chained comparison); `test_shape_continuous_at_branch_boundaries` guards it.
- Quirks inherited from the references are documented in `model.py` comments.
