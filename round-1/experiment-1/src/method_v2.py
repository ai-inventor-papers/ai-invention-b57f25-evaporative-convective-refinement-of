#!/usr/bin/env python3
"""
Implement and compare Newton's Law of Cooling with a Surface-Coupled Fluid Cooling (SCFC) model
against coffee cooling datasets from Manitoba and Arizona.

Newton's Law: dT/dt = -k(T - T_amb)
SCFC Model: dT/dt = -(Q_conv + Q_evap) / (m * c_p)
  where Q_conv = h0 * (T - T_amb)^(1/3) * A * (T - T_amb)  [natural convection]
        Q_evap = m_evap_rate * (T - T_amb) * L_v          [evaporation]
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
L_V = 2260000  # Latent heat of vaporization of water, J/kg
C_P_WATER = 4186  # Specific heat of water, J/(kg*K)
RHO_WATER = 1000  # Density of water, kg/m^3

# Default parameters for coffee cup
VOLUME = 0.3  # liters (300 ml)
MASS = VOLUME * RHO_WATER  # kg
# Assume cylindrical cup: radius ~ 0.04m, height ~ 0.1m
RADIUS = 0.04  # m
HEIGHT = 0.1  # m
SURFACE_AREA = np.pi * RADIUS**2 + 2 * np.pi * RADIUS * HEIGHT  # top + sides

# Experimental datasets
DATASETS = {
    "manitoba_coffee": {
        "t_amb": 21.67,  # 71°F in °C
        "times": np.array([0, 1, 2, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48]) * 60,  # convert to seconds
        "temps": np.array([186, 179, 175, 171, 156, 144, 135, 127, 120, 116, 111, 107, 104]),
        "temp_unit": "F"
    },
    "arizona_coffee": {
        "t_amb": 24.0,  # °C
        "times": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 60,  # seconds
        "temps": np.array([82.3, 79.7, 77.4, 75.1, 73.2, 70.8, 69.3, 67.0, 65.4, 63.4, 62.1]),
        "temp_unit": "C"
    }
}


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32


def newton_cooling(t, T, k, T_amb):
    """Newton's Law of Cooling: dT/dt = -k * (T - T_amb)"""
    return -k * (T - T_amb)


def scfc_cooling(t, T, h0, m_evap_rate, T_amb, m, cp, A):
    """
    Surface-Coupled Fluid Cooling (SCFC) model:
    dT/dt = -(Q_conv + Q_evap) / (m * cp)
    where:
      Q_conv = h0 * (T - T_amb)^(1/3) * A * (T - T_amb) = h0 * A * (T - T_amb)^(4/3)
      Q_evap = m_evap_rate * (T - T_amb) * L_v
    """
    delta_T = T - T_amb
    if delta_T <= 0:
        return 0.0
    
    # Natural convection: h = h0 * (T - T_amb)^(1/3)
    # Q_conv = h * A * (T - T_amb) = h0 * A * (T - T_amb)^(4/3)
    Q_conv = h0 * A * (delta_T ** (4/3))
    
    # Evaporation: Q_evap = m_dot * L_v
    # m_dot = m_evap_rate * (T - T_amb)
    Q_evap = m_evap_rate * delta_T * L_V
    
    return -(Q_conv + Q_evap) / (m * cp)


def solve_newton(t_eval, T0, k, T_amb):
    """Solve Newton's law ODE."""
    sol = solve_ivp(
        newton_cooling, 
        [t_eval[0], t_eval[-1]], 
        [T0], 
        args=(k, T_amb),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    return sol.y[0]


def solve_scfc(t_eval, T0, h0, m_evap_rate, T_amb, m, cp, A):
    """Solve SCFC model ODE."""
    sol = solve_ivp(
        scfc_cooling,
        [t_eval[0], t_eval[-1]],
        [T0],
        args=(h0, m_evap_rate, T_amb, m, cp, A),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    return sol.y[0]


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def fit_newton(times, temps, T_amb):
    """Fit Newton's cooling model to data."""
    T0 = temps[0]
    
    def objective(k):
        if k <= 0:
            return 1e10
        T_pred = solve_newton(times, T0, k[0], T_amb)
        return compute_rmse(temps, T_pred)
    
    # Initial guess for k from linear regression on log delta_T
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
    """Fit SCFC model to data using global optimization then local refinement."""
    T0 = temps[0]
    
    def objective(params):
        h0, m_evap_rate = params
        if h0 <= 0 or m_evap_rate < 0:
            return 1e10
        T_pred = solve_scfc(times, T0, h0, m_evap_rate, T_amb, m, cp, A)
        return compute_rmse(temps, T_pred)
    
    # Initial guesses
    delta_T0 = T0 - T_amb
    k_newton = 1e-3
    h0_guess = k_newton * m * cp / (A * delta_T0**(1/3))
    h0_guess = max(h0_guess, 1e-2)
    m_evap_guess = 1e-6
    
    # First, use differential evolution for global search
    bounds = [(1e-2, 1000.0), (0.0, 1e-2)]
    result_de = differential_evolution(
        objective, 
        bounds, 
        maxiter=50, 
        popsize=10,
        seed=42,
        atol=1e-6
    )
    
    # Then refine with L-BFGS-B
    result = minimize(
        objective, 
        result_de.x, 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    return result.x[0], result.x[1], result.fun


def run_model_comparison():
    """Run model fitting and comparison on both datasets."""
    results = {}
    
    for dataset_name, dataset in DATASETS.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        T_amb = dataset["t_amb"]
        times = dataset["times"]
        temps = dataset["temps"]
        temp_unit = dataset["temp_unit"]
        
        # Convert temperatures to Celsius for modeling
        if temp_unit == "F":
            temps_c = fahrenheit_to_celsius(temps)
            T0_c = temps_c[0]
        else:
            temps_c = temps
            T0_c = temps_c[0]
        
        logger.info(f"  T_amb = {T_amb:.2f}°C, T0 = {T0_c:.2f}°C")
        logger.info(f"  Time range: {times[0]/60:.1f} - {times[-1]/60:.1f} min")
        logger.info(f"  Data points: {len(times)}")
        
        # Fit Newton's model
        logger.info("  Fitting Newton's Law of Cooling...")
        k_newton, newton_rmse = fit_newton(times, temps_c, T_amb)
        T_pred_newton = solve_newton(times, T0_c, k_newton, T_amb)
        newton_mae = compute_mae(temps_c, T_pred_newton)
        logger.info(f"    k = {k_newton:.6f} 1/s, RMSE = {newton_rmse:.4f}°C, MAE = {newton_mae:.4f}°C")
        
        # Fit SCFC model
        logger.info("  Fitting SCFC model...")
        h0, m_evap_rate, scfc_rmse = fit_scfc(times, temps_c, T_amb, MASS, C_P_WATER, SURFACE_AREA)
        T_pred_scfc = solve_scfc(times, T0_c, h0, m_evap_rate, T_amb, MASS, C_P_WATER, SURFACE_AREA)
        scfc_mae = compute_mae(temps_c, T_pred_scfc)
        logger.info(f"    h0 = {h0:.4f}, m_evap_rate = {m_evap_rate:.2e} kg/(s*K)")
        logger.info(f"    RMSE = {scfc_rmse:.4f}°C, MAE = {scfc_mae:.4f}°C")
        
        # Residuals
        residuals_newton = temps_c - T_pred_newton
        residuals_scfc = temps_c - T_pred_scfc
        
        # Compute residual statistics
        rms_newton = np.sqrt(np.mean(residuals_newton**2))
        rms_scfc = np.sqrt(np.mean(residuals_scfc**2))
        bias_newton = np.mean(residuals_newton)
        bias_scfc = np.mean(residuals_scfc)
        
        # Convert predictions back to original units for output
        if temp_unit == "F":
            T_pred_newton_out = celsius_to_fahrenheit(T_pred_newton)
            T_pred_scfc_out = celsius_to_fahrenheit(T_pred_scfc)
            temps_out = temps
        else:
            T_pred_newton_out = T_pred_newton
            T_pred_scfc_out = T_pred_scfc
            temps_out = temps
        
        results[dataset_name] = {
            "dataset": dataset_name,
            "t_amb": T_amb,
            "times_s": times.tolist(),
            "times_min": (times / 60).tolist(),
            "temps_observed": temps_out.tolist(),
            "temps_observed_c": temps_c.tolist(),
            "newton": {
                "k": float(k_newton),
                "rmse_c": float(newton_rmse),
                "mae_c": float(newton_mae),
                "temps_pred_c": T_pred_newton.tolist(),
                "temps_pred_original": T_pred_newton_out.tolist(),
                "residuals_c": residuals_newton.tolist(),
                "residual_rms": float(rms_newton),
                "residual_bias": float(bias_newton)
            },
            "scfc": {
                "h0": float(h0),
                "m_evap_rate": float(m_evap_rate),
                "rmse_c": float(scfc_rmse),
                "mae_c": float(scfc_mae),
                "temps_pred_c": T_pred_scfc.tolist(),
                "temps_pred_original": T_pred_scfc_out.tolist(),
                "residuals_c": residuals_scfc.tolist(),
                "residual_rms": float(rms_scfc),
                "residual_bias": float(bias_scfc)
            },
            "comparison": {
                "rmse_improvement": float((newton_rmse - scfc_rmse) / newton_rmse * 100),
                "mae_improvement": float((newton_mae - scfc_mae) / newton_mae * 100)
            }
        }
    
    return results


def generate_synthetic_test():
    """Generate synthetic data from SCFC model and test parameter recovery."""
    logger.info("Generating synthetic test data...")
    
    # True parameters
    T_amb = 25.0
    T0 = 90.0
    h0_true = 10.0
    m_evap_true = 1e-5
    times = np.linspace(0, 30 * 60, 31)  # 30 minutes, 1 min intervals
    
    # Generate true curve
    T_true = solve_scfc(times, T0, h0_true, m_evap_true, T_amb, MASS, C_P_WATER, SURFACE_AREA)
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(T_true))
    T_noisy = T_true + noise
    
    # Fit SCFC model to noisy data
    h0_fit, m_evap_fit, rmse = fit_scfc(times, T_noisy, T_amb, MASS, C_P_WATER, SURFACE_AREA)
    
    # Recovery accuracy
    h0_error = abs(h0_fit - h0_true) / h0_true * 100
    m_evap_error = abs(m_evap_fit - m_evap_true) / m_evap_true * 100 if m_evap_true > 0 else float('inf')
    
    logger.info(f"  True: h0={h0_true}, m_evap_rate={m_evap_true}")
    logger.info(f"  Fit:  h0={h0_fit:.4f}, m_evap_rate={m_evap_fit:.2e}")
    logger.info(f"  Errors: h0={h0_error:.1f}%, m_evap_rate={m_evap_error:.1f}%")
    logger.info(f"  RMSE on noisy data: {rmse:.4f}°C")
    
    return {
        "true_params": {"h0": h0_true, "m_evap_rate": m_evap_true},
        "fit_params": {"h0": float(h0_fit), "m_evap_rate": float(m_evap_fit)},
        "errors_pct": {"h0": float(h0_error), "m_evap_rate": float(m_evap_error)},
        "fit_rmse": float(rmse),
        "times_s": times.tolist(),
        "temps_true": T_true.tolist(),
        "temps_noisy": T_noisy.tolist()
    }


@logger.catch(reraise=True)
def main():
    logger.info("Starting SCFC vs Newton's Law comparison experiment")
    
    # Run synthetic test first
    synthetic_result = generate_synthetic_test()
    
    # Run on real datasets
    results = run_model_comparison()
    
    # Prepare output in exp_gen_sol_out format
    output = {
        "metadata": {
            "method_name": "SCFC_vs_Newton",
            "description": "Comparison of Newton's Law of Cooling and Surface-Coupled Fluid Cooling model on coffee cooling datasets",
            "parameters": {
                "volume_l": VOLUME,
                "mass_kg": MASS,
                "radius_m": RADIUS,
                "height_m": HEIGHT,
                "surface_area_m2": SURFACE_AREA,
                "cp_water": C_P_WATER,
                "L_v": L_V,
                "rho_water": RHO_WATER
            }
        },
        "datasets": [
            {
                "dataset": "synthetic_test",
                "examples": [
                    {
                        "input": "Parameter recovery test on synthetic SCFC data",
                        "output": json.dumps(synthetic_result),
                        "metadata_fit_rmse": synthetic_result["fit_rmse"]
                    }
                ]
            },
            {
                "dataset": "manitoba_coffee",
                "examples": [
                    {
                        "input": "Fit cooling models to Manitoba coffee experiment data (71°F ambient)",
                        "output": json.dumps(results["manitoba_coffee"]),
                        "metadata_newton_rmse_c": results["manitoba_coffee"]["newton"]["rmse_c"],
                        "metadata_scfc_rmse_c": results["manitoba_coffee"]["scfc"]["rmse_c"],
                        "metadata_rmse_improvement_pct": results["manitoba_coffee"]["comparison"]["rmse_improvement"]
                    }
                ]
            },
            {
                "dataset": "arizona_coffee",
                "examples": [
                    {
                        "input": "Fit cooling models to Arizona coffee experiment data (24°C ambient)",
                        "output": json.dumps(results["arizona_coffee"]),
                        "metadata_newton_rmse_c": results["arizona_coffee"]["newton"]["rmse_c"],
                        "metadata_scfc_rmse_c": results["arizona_coffee"]["scfc"]["rmse_c"],
                        "metadata_rmse_improvement_pct": results["arizona_coffee"]["comparison"]["rmse_improvement"]
                    }
                ]
            }
        ]
    }
    
    # Save full results
    output_path = Path("method_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {output_path}")
    
    # Print summary
    logger.info("\n=== SUMMARY ===")
    for ds_name in ["manitoba_coffee", "arizona_coffee"]:
        r = results[ds_name]
        logger.info(f"\n{ds_name}:")
        logger.info(f"  Newton:  RMSE={r['newton']['rmse_c']:.4f}°C, MAE={r['newton']['mae_c']:.4f}°C, k={r['newton']['k']:.6f} 1/s")
        logger.info(f"  SCFC:    RMSE={r['scfc']['rmse_c']:.4f}°C, MAE={r['scfc']['mae_c']:.4f}°C, h0={r['scfc']['h0']:.4f}, m_evap={r['scfc']['m_evap_rate']:.2e}")
        logger.info(f"  Improvement: RMSE={r['comparison']['rmse_improvement']:.1f}%, MAE={r['comparison']['mae_improvement']:.1f}%")


if __name__ == "__main__":
    main()
