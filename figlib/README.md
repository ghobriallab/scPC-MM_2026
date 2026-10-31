# figlib

Modules shared by the figure scripts.

| Module | Content |
|--------|---------|
| `figstyle.py` | Nature-format style (Arial 5-7 pt, line widths); italic symbols and exponents as separate text runs |
| `palettes.py` | Color-blind-safe palettes; `python3 figlib/palettes.py --check` simulates color-vision deficiencies |
| `layout.py` | Figures and panels laid out in millimetres; saving the SVG and PDF |
| `plots.py` | Box plots with n and brackets, UMAPs, heatmaps, Kaplan-Meier curves, forest plots, bars, keys |
| `icons.py` | Vector icons and the schematics of Figs. 1A and 4A |
| `checks.py` | Automatic checks of every figure (fonts, sizes, overlaps, files) |
| `names.py` | Display names of the signatures and classifier features |
| `targets.py` | Therapeutic targets and per-tumor differential expression |
| `stage_tests.py` | Per-patient units and the stage tests behind the brackets |
| `paths.py` | Where the figure scripts read and write (from `config.py`) |
