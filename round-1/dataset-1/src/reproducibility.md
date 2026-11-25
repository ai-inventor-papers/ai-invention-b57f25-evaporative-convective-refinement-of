# Reproducibility Instructions

## 1. Getting This Artifact

Clone the published GitHub repository and `cd` into this artifact's folder:

```bash
git clone <repo_url>
cd <repo_root>/gen_art_dataset_1
```

## 2. System Requirements and Environment Setup

**Python version:** 3.12 (required)

**System packages (Ubuntu):**
```bash
sudo apt-get install python3.12 python3.12-venv python3.12-dev
```

**Virtual environment creation:**
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install pinned dependencies
cd gen_art_dataset_1
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -r <(python -c "
import tomllib
with open('pyproject.toml','rb') as f:
    data = tomllib.load(f)
print('\n'.join(data['project']['dependencies']))
")
```

All 33 dependencies are pinned in `pyproject.toml` — see that file for exact versions (loguru==0.8.0, requests==2.34.2, jsonschema==4.26.0, numpy==2.5.5, etc.).

## 3. Data and Model Downloads

**No external data or model downloads required.** This artifact generates all data synthetically from first-principles physics equations. No API keys, environment variables, or network access are needed.

The physics parameters are hardcoded in `data.py` (derived from Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*):
- Churchill-Chu natural convection correlation
- Stefan-Boltzmann radiation (σ = 5.67×10⁻⁸ W/m²/K⁴, ε_water = 0.96)
- ASHRAE evaporation correlation (latent heat H_vap = 2.43×10⁶ J/kg)

## 4. Exact Commands Run

**Hardware:** CPU-only (2 vCPUs, 16 GB RAM), no GPU. Runtime: ~2 seconds for data generation.

**Seed:** `random.seed(42)` hardcoded in `data.py` for reproducible Gaussian noise.

```bash
# Step 1: Generate the dataset
source .venv/bin/activate
python data.py
```

This single command produces:
- `full_data_out.json` — 5 datasets, 845 total examples
- `mini_data_out.json` — 1 dataset, 3 examples
- `preview_data_out.json` — 5 datasets, 3 examples each

**Step 2: Validate schema (optional)**
```bash
# Requires jsonschema (included in pyproject.toml dependencies)
python -c "
import json, jsonschema
schema = json.load(open('/Users/Shared/aii_data/.claude/skills/aii-json/schemas/exp_sel_data_out.json'))
data = json.load(open('full_data_out.json'))
validator = jsonschema.Draft7Validator(schema)
errors = list(validator.iter_errors(data))
assert not errors, f'Schema errors: {errors}'
print('Schema validation PASSED')
"
```

**Step 3: Regenerate mini/preview from full (if needed)**
```bash
python -c "
import json
from pathlib import Path
data = json.load(open('full_data_out.json'))
mini = {'metadata': data.get('metadata', {}), 'datasets': [{'dataset': data['datasets'][0]['dataset'], 'examples': data['datasets'][0]['examples'][:3]}]}
Path('mini_data_out.json').write_text(json.dumps(mini, indent=2))
def trunc(obj, maxlen=200):
    if isinstance(obj, str): return obj[:maxlen]+'...' if len(obj)>maxlen else obj
    if isinstance(obj, list): return [trunc(x, maxlen) for x in obj]
    if isinstance(obj, dict): return {k: trunc(v, maxlen) for k, v in obj.items()}
    return obj
preview = {'metadata': trunc(data.get('metadata', {})), 'datasets': [{'dataset': ds['dataset'], 'examples': trunc(ds['examples'][:3])} for ds in data['datasets']]}
Path('preview_data_out.json').write_text(json.dumps(preview, indent=2))
"
```

## 5. Expected Output Files and Numbers

| File | Size | Content |
|------|------|---------|
| `full_data_out.json` | 775K | 5 datasets, 845 examples (exp_sel_data_out schema) |
| `mini_data_out.json` | 3.5K | 1 dataset, 3 examples |
| `preview_data_out.json` | 15K | 5 datasets, 3 examples each (strings truncated to 200 chars) |

**Per-dataset example counts:**
- `ceramic_mug_hot_water_room`: 181 examples
- `glass_beaker_hot_water_room`: 181 examples
- `ceramic_mug_coffee_room`: 121 examples
- `steel_cup_hot_water_humid`: 181 examples
- `wide_bowl_hot_water_room`: 181 examples

**Key numbers a reader should verify:**
- Newton's law RMSE vs measured: ~1.08°C
- SCFC model RMSE vs measured: ~0.14°C
- Newton-SCFC temperature divergence at 30 min: ~1.5°C
- Fitted k values: 0.0002–0.0004 s⁻¹ (condition-dependent)

These appear in `full_data_out.json` → `metadata` field and in each example's `metadata_scfc_predicted_C` vs `metadata_newton_predicted_C` fields.

## 6. Portability Notes

- All paths in `data.py` are anchored on `Path(__file__).parent` (portable).
- No user-uploaded inputs are used.
- No sibling artifact inputs are needed.
- The `scratch/` directory and `logs/` directory are auto-created by `data.py` and are marked for deletion in `.aii/manifest.yaml` (regenerable).
