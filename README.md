# Cutting Thermal Model

Peclet-normalized tool-chip contact temperature model with 2D subsurface
temperature and residual stress, shown in a Streamlit UI.

```
uv sync
uv run python app.py     # launches the UI in the browser
uv run pytest            # tests
```

## Files (all at root)
- `model.py` — all physics: materials, Kienzle force fit, Peclet number, flash temperature, surface shape, subsurface field, residual stress, `solve()` / `solve_2d()`
- `app.py` — the whole UI
- `test_model.py` — tests
- `ref/` — source papers, slides, spreadsheets

## Notes
- Fixed a dead-branch bug from `ref/Subsurface_thermal.m` line 52 (chained comparison); `test_shape_continuous_at_branch_boundaries` guards it.
- AA6061 thermal properties (and several AA7050/AA6061 stress fits) in the source workbook are copies of other materials' — treat as approximate.
- Kienzle force-from-feed only exists for Ti-6Al-4V; other materials use direct Fc.
- Quirks inherited from the references are documented in `model.py` comments.
