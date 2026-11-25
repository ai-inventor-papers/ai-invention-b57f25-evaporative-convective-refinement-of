# Cooling Curve Benchmark Dataset

Benchmark dataset for comparing Newton's law of cooling against the SCFC (Stefan-Cooling-Film-Correction) model.

## What This Is

5 synthetic cooling curves recording temperature-vs-time data for open-surface liquids cooling from ~85-95°C over 30 minutes. Each curve includes measured temperatures (with Gaussian noise), SCFC model predictions, and Newton's law predictions. Conditions vary across container material (ceramic, glass, steel, wide ceramic), fluid type (water, coffee), and ambient humidity (25-70%).

## Layout

| File | Description |
|------|-------------|
| `full_data_out.json` (775K) | Complete dataset: 5 curves, 845 examples |
| `mini_data_out.json` (3.5K) | First curve, 3 examples |
| `preview_data_out.json` (15K) | All 5 curves, 3 examples each, strings truncated |
| `data.py` | Self-contained generator script |
| `reproducibility.md` | Step-by-step reproduction guide |
| `pyproject.toml` | Pinned dependencies |

## Datasets (5)

| Dataset | Points | Container | Fluid | RH | T_initial | T_amb |
|---------|--------|-----------|-------|-----|-----------|-------|
| ceramic_mug_hot_water_room | 181 | ceramic mug | water | 45% | 92°C | 22°C |
| glass_beaker_hot_water_room | 181 | glass beaker | water | 50% | 95°C | 23°C |
| ceramic_mug_coffee_room | 121 | ceramic mug | coffee | 40% | 88°C | 21°C |
| steel_cup_hot_water_humid | 181 | steel cup | water | 70% | 90°C | 24°C |
| wide_bowl_hot_water_room | 181 | wide ceramic bowl | water | 48% | 93°C | 22°C |

## Statistics

- Newton's law RMSE vs measured: ~1.08°C
- SCFC model RMSE vs measured: ~0.14°C
- Newton-SCFC divergence at 30 min: ~1.5°C
- Fitted cooling constant k: 0.0002–0.0004 s⁻¹

## How to Run

```bash
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -r <(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print('\n'.join(d['project']['dependencies']))")
python data.py
```

## Sources

- **Synthetic curves**: First-principles physics from Incropera & DeWitt (*Fundamentals of Heat and Mass Transfer*) — Churchill-Chu natural convection, Stefan-Boltzmann radiation, ASHRAE evaporation correlations. Gaussian measurement noise σ=0.10–0.20°C. Seed: 42.
- No external data downloads required.

## Restoring Removed Files

- `.venv/`: `uv venv .venv --python=3.12 && uv pip install -r pyproject.toml`
- `scratch/`, `logs/`: Auto-created by `python data.py`
