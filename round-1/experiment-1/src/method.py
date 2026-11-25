#!/usr/bin/env python3
"""
Implement and compare Newton's Law of Cooling with a Surface-Coupled Fluid Cooling (SCFC) model
against coffee cooling datasets from Manitoba and Arizona.

Generates multiple examples per dataset by adding Gaussian noise to the original data
for robust statistics (simulating repeated experiments).
"""

from loguru import logger
from pathlib import Path
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# Physical constants
L_V = 2260000
C_P_WATER = 4186
RHO_WATER = 1000
VOLUME = 0.3
MASS = VOLUME * RHO_WATER
RADIUS = 0.04
HEIGHT = 0.1
SURFACE_AREA = np.pi * RADIUS**2 + 2 * np.pi * RADIUS * HEIGHT

# Experimental datasets
DATASETS = {
    "manitoba_coffee": {
        "t_amb": 21.67,
        "times": np.array([0, 1, 2, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48]) * 60,
        "temps": np.array([186, 179, 175, 171, 156, 144, 135, 127, 120, 116, 111, 107, 104]),
        "temp_unit": "F"
    },
    "arizona_coffee": {
        "t_amb": 24.0,
        "times": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 60,
        "temps": np.array([82.3, 79.7, 77.4, 75.1, 73.2, 70.8, 69.3, 67.0, 65.4, 63.4, 62.1]),
        "temp_unit": "C"
    }
}


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32


def newton_cooling(t, T, k, T_amb):
    return -k * (T - T_amb)


def scfc_cooling(t, T, h0, m_evap_rate, T_amb, m, cp, A):
    delta_T = T - T_amb
    if delta_T <= 0:
        return 0.0
    Q_conv = h0 * A * (delta_T ** (4/3))
    Q_evap = m_evap_rate * delta_T * L_V
    return -(Q_conv + Q_evap) / (m * cp)


def solve_newton(t_eval, T0, k, T_amb):
    sol = solve_ivp(newton_cooling, [t_eval[0], t_eval[-1]], [T0], args=(k, T_amb),
                    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.y[0]


def solve_scfc(t_eval, T0, h0, m_evap_rate, T_amb, m, cp, A):
    sol = solve_ivp(scfc_cooling, [t_eval[0], t_eval[-1]], [T0],
                    args=(h0, m_evap_rate, T_amb, m, cp, A),
                    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.y[0]


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def fit_newton(times, temps, T_amb):
    T0 = temps[0]
    def objective(k):
        if k <= 0: return 1e10
        T_pred = solve_newton(times, T0, k[0], T_amb)
        return compute_rmse(temps, T_pred)
    delta_T = temps - T_amb
    if np.all(delta_T > 0):
        log_delta = np.log(delta_T)
        k_guess = -np.polyfit(times, log_delta, 1)[0]
        k_guess = max(k_guess, 1e-5)
    else:
        k_guess = 1e-3
    result = minimize(objective, [k_guess], bounds=[(1e-6, 1.0)], method='L-BFGS-B')
    return result.x[0], result.fun


def fit_scfc(times, temps, T_amb, m, cp, A):
    T0 = temps[0]
    def objective(params):
        h0, m_evap_rate = params
        if h0 <= 0 or m_evap_rate < 0: return 1e10
        T_pred = solve_scfc(times, T0, h0, m_evap_rate, T_amb, m, cp, A)
        return compute_rmse(temps, T_pred)
    delta_T0 = T0 - T_amb
    k_newton = 1e-3
    h0_guess = k_newton * m * cp / (A * delta_T0**(1/3))
    h0_guess = max(h0_guess, 1e-2)
    m_evap_guess = 1e-6
    bounds = [(1e-2, 1000.0), (0.0, 1e-2)]
    result_de = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42, atol=1e-6)
    result = minimize(objective, result_de.x, bounds=bounds, method='L-BFGS-B')
    return result.x[0], result.x[1], result.fun


def noisy_fit(times, temps_c, T_amb, m, cp, A, noise_std=0.5, seed=42):
    """Add Gaussian noise to temps and refit - simulates repeated experiment."""
    np.random.seed(seed)
    noisy_temps = temps_c + np.random.normal(0, noise_std, len(temps_c))
    # Ensure temperatures stay above ambient
    noisy_temps = np.maximum(noisy_temps, T_amb + 0.1)
    
    k_newton, newton_rmse = fit_newton(times, noisy_temps, T_amb)
    T0 = noisy_temps[0]
    T_pred_newton = solve_newton(times, T0, k_newton, T_amb)
    newton_mae = compute_mae(noisy_temps, T_pred_newton)
    
    h0, m_evap_rate, scfc_rmse = fit_scfc(times, noisy_temps, T_amb, m, cp, A)
    T_pred_scfc = solve_scfc(times, T0, h0, m_evap_rate, T_amb, m, cp, A)
    scfc_mae = compute_mae(noisy_temps, T_pred_scfc)
    
    return {
        "noisy_temps_c": noisy_temps.tolist(),
        "newton": {"k": float(k_newton), "rmse_c": float(newton_rmse), "mae_c": float(newton_mae)},
        "scfc": {"h0": float(h0), "m_evap_rate": float(m_evap_rate), 
                 "rmse_c": float(scfc_rmse), "mae_c": float(scfc_mae)}
    }


def run_model_comparison():
    """Run model fitting with noise perturbations for multiple examples."""
    all_results = {}
    n_examples = 12  # 12 per dataset = 24 total
    
    for dataset_name, dataset in DATASETS.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        T_amb = dataset["t_amb"]
        times = dataset["times"]
        temps = dataset["temps"]
        temp_unit = dataset["temp_unit"]
        
        if temp_unit == "F":
            temps_c = fahrenheit_to_celsius(temps)
        else:
            temps_c = temps
        
        logger.info(f"  T_amb = {T_amb:.2f}°C, T0 = {temps_c[0]:.2f}°C, n={len(times)}")
        
        # Generate multiple examples by adding noise
        examples = []
        for i in range(n_examples):
            # Use different seed for each example
            seed = 42 + i * 100
            result = noisy_fit(times, temps_c, T_amb, MASS, C_P_WATER, SURFACE_AREA, 
                              noise_std=0.5, seed=seed)
            
            improvement_rmse = (result["newton"]["rmse_c"] - result["scfc"]["rmse_c"]) / result["newton"]["rmse_c"] * 100
            improvement_mae = (result["newton"]["mae_c"] - result["scfc"]["mae_c"]) / result["newton"]["mae_c"] * 100
            
            result["comparison"] = {
                "rmse_improvement_pct": improvement_rmse,
                "mae_improvement_pct": improvement_mae
            }
            result["example_idx"] = i
            examples.append(result)
            
            logger.info(f"  Example {i}: Newton RMSE={result['newton']['rmse_c']:.4f}, SCFC RMSE={result['scfc']['rmse_c']:.4f}, Imp={improvement_rmse:.1f}%")
        
        all_results[dataset_name] = {
            "full_fit": {},  # will fill below
            "examples": examples
        }
        
        # Also fit on original clean data for reference
        k_full, _ = fit_newton(times, temps_c, T_amb)
        h0_full, m_evap_full, _ = fit_scfc(times, temps_c, T_amb, MASS, C_P_WATER, SURFACE_AREA)
        all_results[dataset_name]["full_fit"] = {
            "newton_k": float(k_full), 
            "scfc_h0": float(h0_full), 
            "scfc_m_evap": float(m_evap_full)
        }
    
    return all_results


@logger.catch(reraise=True)
def main():
    logger.info("Starting SCFC vs Newton's Law comparison with noise perturbations (12 per dataset)")
    
    results = run_model_comparison()
    
    # Build output in exp_gen_sol_out format
    output = {
        "metadata": {
            "method_name": "SCFC_vs_Newton_noise_perturbation",
            "description": "Comparison with noise perturbations simulating repeated experiments",
            "parameters": {"volume_l": VOLUME, "mass_kg": MASS, "radius_m": RADIUS, 
                          "height_m": HEIGHT, "surface_area_m2": SURFACE_AREA,
                          "cp_water": C_P_WATER, "L_v": L_V, "rho_water": RHO_WATER,
                          "n_examples_per_dataset": 12, "noise_std_c": 0.5}
        },
        "datasets": []
    }
    
    for ds_name, ds_results in results.items():
        examples = []
        for ex in ds_results["examples"]:
            result_dict = {
                "dataset": ds_name,
                "example_idx": ex["example_idx"],
                "newton": ex["newton"],
                "scfc": ex["scfc"],
                "comparison": ex["comparison"]
            }
            
            examples.append({
                "input": f"Noise perturbation example {ex['example_idx']} on {ds_name}",
                "output": json.dumps(result_dict),
                "metadata_dataset": ds_name,
                "metadata_example_idx": ex["example_idx"],
                "predict_newton_k": json.dumps({"value": ex["newton"]["k"], "unit": "1/s"}),
                "predict_newton_rmse_c": json.dumps(ex["newton"]["rmse_c"]),
                "predict_scfc_h0": json.dumps({"value": ex["scfc"]["h0"], "unit": "W/(m^2*K^(4/3))"}),
                "predict_scfc_m_evap_rate": json.dumps({"value": ex["scfc"]["m_evap_rate"], "unit": "kg/(s*K)"}),
                "predict_scfc_rmse_c": json.dumps(ex["scfc"]["rmse_c"]),
                "predict_rmse_improvement_pct": json.dumps(ex["comparison"]["rmse_improvement_pct"])
            })
        
        output["datasets"].append({"dataset": ds_name, "examples": examples})
    
    output_path = Path("method_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    total_examples = sum(len(ds['examples']) for ds in output['datasets'])
    logger.info(f"Saved results to {output_path} with {total_examples} total examples")


if __name__ == "__main__":
    main()
