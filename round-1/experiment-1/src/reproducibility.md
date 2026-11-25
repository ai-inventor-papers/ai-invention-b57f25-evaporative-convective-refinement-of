# Reproducibility Guide: SCFC vs Newton's Law of Cooling Comparison

This document provides complete step-by-step instructions to reproduce the exact results from this experiment on Ubuntu.

## 1. Getting This Artifact

This workspace is published as one folder of a public GitHub repository. To reproduce:

```bash
git clone <repository-url>
cd <repository-root>/round-1/experiment-1/src
```

All paths in this guide are relative to this folder.

## 2. System Requirements

### System Packages (Ubuntu)
```bash
sudo apt-get update && sudo apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    build-essential pkg-config \
    libfreetype6-dev libpng-dev
```

### Python Version
- **Python 3.12** (tested with 3.12.14)

## 3. Environment Setup

### Create Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies (Exact Pinned Versions)
```bash
pip install --upgrade pip
pip install \
    contourpy==1.4.0 \
    cycler==0.12.1 \
    fonttools==4.68.0 \
    kiwisolver==1.5.1 \
    loguru==0.8.0 \
    matplotlib==3.11.2 \
    numpy==2.5.5 \
    packaging==26.4 \
    pillow==13.0.0 \
    pyparsing==3.3.3 \
    python-dateutil==4.0.0 \
    scipy==1.18.1 \
    tzdata==2026.6
```

Or use the pyproject.toml:
```bash
pip install -e .
```

### Verify Installation
```bash
python -c "import numpy, scipy, matplotlib, loguru; print('All imports OK')"
python -c "import numpy; print(f'numpy={numpy.__version__}')"
```

## 4. Data Files

The experiment uses two experimental coffee cooling datasets included in this workspace:

- `data_manitoba.json` - University of Manitoba coffee cooling data (71°F ambient)
- `data_arizona.json` - University of Arizona coffee cooling data (24°C ambient)

These are **included in the repository** and require no external downloads.

## 5. Running the Experiment

### Commands (in order)
```bash
# 1. Ensure you're in the experiment folder
cd round-1/experiment-1/src

# 2. Activate virtual environment (if not already active)
source .venv/bin/activate

# 3. Run the main experiment script
python method.py
```

### Runtime & Hardware
- **Hardware**: CPU-only (no GPU required)
- **Approximate runtime**: 30-60 seconds
- **Random seed**: 42 (set in differential_evolution for reproducibility)
- **Memory usage**: < 100 MB

### Expected Output
The script will:
1. Run synthetic parameter recovery test
2. Fit both models to Manitoba coffee data
3. Fit both models to Arizona coffee data
4. Save results to `method_out.json`
5. Print summary to console

## 6. Expected Results

### Console Output Summary
```
=== SUMMARY ===

manitoba_coffee:
  Newton:  RMSE=2.1255°C, MAE=1.8977°C, k=0.000497 1/s
  SCFC:    RMSE=1.9748°C, MAE=1.7609°C, h0=993.7573, m_evap=2.30e-04
  Improvement: RMSE=7.1%, MAE=7.2%

arizona_coffee:
  Newton:  RMSE=0.1573°C, MAE=0.1417°C, k=0.000717 1/s
  SCFC:    RMSE=0.1530°C, MAE=0.1353°C, h0=999.9377, m_evap=3.49e-04
  Improvement: RMSE=2.7%, MAE=4.5%
```

### Output Files Generated
| File | Description |
|------|-------------|
| `method_out.json` | Full results in exp_gen_sol_out schema format |
| `full_method_out.json` | Identical to method_out.json |
| `mini_method_out.json` | First 3 examples only |
| `preview_method_out.json` | Mini with strings truncated to 200 chars |
| `figures/*.pdf` | Publication-quality vector figures |
| `figures/*.png` | Raster versions of figures |

### Key Metrics to Verify
From `method_out.json` (or `full_method_out.json`):

**Manitoba Coffee:**
- Newton RMSE: 2.1255°C, MAE: 1.8977°C, k: 4.97e-4 1/s
- SCFC RMSE: 1.9748°C, MAE: 1.7609°C, h0: 993.8, m_evap: 2.30e-4 kg/(s·K)
- RMSE improvement: 7.1%, MAE improvement: 7.2%

**Arizona Coffee:**
- Newton RMSE: 0.1573°C, MAE: 0.1417°C, k: 7.17e-4 1/s
- SCFC RMSE: 0.1530°C, MAE: 0.1353°C, h0: 1000.0, m_evap: 3.49e-4 kg/(s·K)
- RMSE improvement: 2.7%, MAE improvement: 4.5%

### Figures Generated
All figures are saved in `figures/`:
- `manitoba_cooling.pdf` / `.png` - Cooling curves comparison
- `arizona_cooling.pdf` / `.png` - Cooling curves comparison
- `manitoba_residuals.pdf` / `.png` - Residuals (observed - predicted)
- `arizona_residuals.pdf` / `.png` - Residuals
- `rmse_comparison.pdf` / `.png` - RMSE bar chart
- `mae_comparison.pdf` / `.png` - MAE bar chart

## 7. Re-generating Figures (Optional)

Figures are generated from spec files using aii-data-fig-gen. To regenerate:

```bash
# Install aii-data-fig-gen dependencies (if not already available)
pip install matplotlib scipy numpy

# Generate each figure
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_cooling_manitoba.json --out figures/manitoba_cooling
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_cooling_arizona.json --out figures/arizona_cooling
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_residuals_manitoba.json --out figures/manitoba_residuals
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_residuals_arizona.json --out figures/arizona_residuals
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_rmse_comparison.json --out figures/rmse_comparison
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec figures/spec_mae_comparison.json --out figures/mae_comparison
```

Note: The aii-data-fig-gen skill is part of the AI Inventor pipeline. If unavailable, the existing PDF/PNG files in `figures/` are the reference outputs.

## 8. Validation

### Schema Validation
```bash
python -c "
import json
from jsonschema import validate
with open('/path/to/aii-json/schemas/exp_gen_sol_out.json') as f:
    schema = json.load(f)
with open('method_out.json') as f:
    data = json.load(f)
validate(instance=data, schema=schema)
print('Schema validation: PASSED')
"
```

### File Sizes
All output files are well under the 100 MB limit:
- `method_out.json`: ~10 KB
- `full_method_out.json`: ~10 KB
- Figures: ~10-100 KB each (PDF), ~50-100 KB each (PNG)

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated and dependencies installed |
| Matplotlib font warnings | "findfont: Failed to find font weight normal for CMU Serif" - harmless, uses fallback |
| Differential evolution warnings | Convergence warnings are normal; results are still valid |
| Long runtime | Ensure you're using the provided seed (42); different seeds may vary slightly |

## 10. Notes

- The experiment is deterministic with the fixed random seed (42)
- No external API keys or downloads required
- All data is embedded in the repository
- The SCFC model fitting uses differential evolution (global) + L-BFGS-B (local refinement)
- Results may vary by < 0.01% across different NumPy/SciPy versions due to floating-point differences
