# SCFC vs Newton's Law of Cooling Comparison

This experiment implements and compares Newton's Law of Cooling (baseline) with a Surface-Coupled Fluid Cooling (SCFC) model against experimental coffee cooling datasets from Manitoba and Arizona.

## Project Layout

```
.
├── method.py                    # Main experiment script
├── method_out.json              # Results in exp_gen_sol_out schema format
├── data_manitoba.json           # Manitoba coffee experiment data (71°F ambient)
├── data_arizona.json            # Arizona coffee experiment data (24°C ambient)
├── logs/run.log                 # Detailed experiment logs
├── figures/
│   ├── manitoba_cooling.pdf     # Cooling curves: observed vs Newton vs SCFC (Manitoba)
│   ├── manitoba_cooling.png
│   ├── arizona_cooling.pdf      # Cooling curves: observed vs Newton vs SCFC (Arizona)
│   ├── arizona_cooling.png
│   ├── manitoba_residuals.pdf   # Residuals: Newton vs SCFC (Manitoba)
│   ├── manitoba_residuals.png
│   ├── arizona_residuals.pdf    # Residuals: Newton vs SCFC (Arizona)
│   ├── arizona_residuals.png
│   ├── rmse_comparison.pdf      # RMSE comparison bar chart
│   ├── rmse_comparison.png
│   ├── mae_comparison.pdf       # MAE comparison bar chart
│   ├── mae_comparison.png
│   └── spec_*.json              # Figure specifications for aii-data-fig-gen
├── .venv/                       # Virtual environment (regenerable)
└── .aii/manifest.yaml           # Artifact manifest
```

## Models

### Newton's Law of Cooling (Baseline)
```
dT/dt = -k(T - T_amb)
```
Single parameter: `k` (cooling rate constant, 1/s)

### Surface-Coupled Fluid Cooling (SCFC) Model
```
dT/dt = -(Q_conv + Q_evap) / (m * c_p)
Q_conv = h0 * A * (T - T_amb)^(4/3)    # Natural convection
Q_evap = m_evap_rate * (T - T_amb) * L_v  # Evaporation
```
Parameters: `h0` (convection coefficient), `m_evap_rate` (evaporation rate coefficient)

## Results Summary

| Dataset | Model | RMSE (°C) | MAE (°C) | Parameters |
|---------|-------|-----------|----------|------------|
| Manitoba Coffee | Newton | 2.1255 | 1.8977 | k = 4.97e-4 1/s |
| Manitoba Coffee | SCFC | **1.9748** | **1.7609** | h0 = 993.8, m_evap = 2.30e-4 kg/(s·K) |
| Arizona Coffee | Newton | 0.1573 | 0.1417 | k = 7.17e-4 1/s |
| Arizona Coffee | SCFC | **0.1530** | **0.1353** | h0 = 1000.0, m_evap = 3.49e-4 kg/(s·K) |

**Improvement:**
- Manitoba: RMSE ↓ 7.1%, MAE ↓ 7.2%
- Arizona: RMSE ↓ 2.7%, MAE ↓ 4.5%

## How to Run

```bash
# Create virtual environment
uv venv .venv --python=3.12

# Install dependencies
uv pip install --python .venv/bin/python numpy scipy matplotlib loguru

# Run experiment
.venv/bin/python method.py
```

## Restoring Removed Files

### Virtual Environment (.venv/)
```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python numpy scipy matplotlib loguru
```

### Figures
Figures are generated from spec files using aii-data-fig-gen:
```bash
.venv/bin/python /Volumes/aii-s20008/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_cooling_manitoba.json --out figures/manitoba_cooling
.venv/bin/python /Volumes/aii-s20008/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_cooling_arizona.json --out figures/arizona_cooling
.venv/bin/python /Volumes/aii-s20008/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_residuals_manitoba.json --out figures/manitoba_residuals
.venv/bin/python /Volumes/aii-s20008/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_residuals_arizona.json --out figures/arizona_residuals
.venv/bin/python /Volumes/aii-s20008/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_rmse_comparison.json --out figures/rmse_comparison
.venv/bin/python /Volumes/aii-s20008/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_mae_comparison.json --out figures/mae_comparison
```

## Data Sources

- **Manitoba Coffee**: University of Manitoba workshop data, room temperature 71°F (21.67°C)
- **Arizona Coffee**: University of Arizona dataset, room temperature 24°C