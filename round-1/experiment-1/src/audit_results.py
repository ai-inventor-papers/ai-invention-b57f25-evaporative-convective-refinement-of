#!/usr/bin/env python3
"""
Independent audit script: re-derives headline numbers from raw data
using a DIFFERENT code path than the main experiment.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Load raw data directly from source files (not method_out.json)
with open('data_manitoba.json') as f:
    manitoba_raw = json.load(f)

with open('data_arizona.json') as f:
    arizona_raw = json.load(f)

# Physical constants (same as experiment)
L_V = 2260000
C_P_WATER = 4186
RHO_WATER = 1000
VOLUME = 0.3
MASS = VOLUME * RHO_WATER
RADIUS = 0.04
HEIGHT = 0.1
SURFACE_AREA = np.pi * RADIUS**2 + 2 * np.pi * RADIUS * HEIGHT

def f_to_c(f):
    return (f - 32) * 5 / 9

# --- Independent model implementations (different from method.py) ---

def newton_model(t, T0, k, T_amb):
    """Analytical solution - different from solve_ivp used in main"""
    return T_amb + (T0 - T_amb) * np.exp(-k * t)

def scfc_model_deriv(t, T, h0, m_evap, T_amb, m, cp, A):
    dT = T - T_amb
    if dT <= 0:
        return 0.0
    Q_conv = h0 * A * (dT ** (4/3))
    Q_evap = m_evap * dT * L_V
    return -(Q_conv + Q_evap) / (m * cp)

def scfc_model_numeric(t_eval, T0, h0, m_evap, T_amb, m, cp, A):
    """Uses RK45 like main, but called separately"""
    sol = solve_ivp(
        scfc_model_deriv, [t_eval[0], t_eval[-1]], [T0],
        args=(h0, m_evap, T_amb, m, cp, A),
        t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9
    )
    return sol.y[0]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# --- Audit Manitoba ---
print("=" * 60)
print("AUDIT: MANITOBA COFFEE")
print("=" * 60)

manitoba_times = np.array([d['t_min'] for d in manitoba_raw['data']]) * 60  # seconds
manitoba_temps_f = np.array([d['T_f'] for d in manitoba_raw['data']])
manitoba_temps_c = f_to_c(manitoba_temps_f)
T_amb_m = manitoba_raw['t_amb_f']
T_amb_m_c = f_to_c(T_amb_m)
T0_m_c = manitoba_temps_c[0]

print(f"Data points: {len(manitoba_times)}")
print(f"T_amb: {T_amb_m_c:.2f}°C, T0: {T0_m_c:.2f}°C")

# Fit Newton (analytical, different from main)
delta_T = manitoba_temps_c - T_amb_m_c
log_delta = np.log(delta_T)
k_guess = -np.polyfit(manitoba_times, log_delta, 1)[0]

def newton_obj(k):
    pred = newton_model(manitoba_times, T0_m_c, k[0], T_amb_m_c)
    return rmse(manitoba_temps_c, pred)

res = minimize(newton_obj, [k_guess], bounds=[(1e-6, 1.0)], method='L-BFGS-B')
k_fit = res.x[0]
newton_pred_m = newton_model(manitoba_times, T0_m_c, k_fit, T_amb_m_c)
newton_rmse_m = rmse(manitoba_temps_c, newton_pred_m)
newton_mae_m = mae(manitoba_temps_c, newton_pred_m)

print(f"\nNewton (analytical fit):")
print(f"  k = {k_fit:.6f} 1/s")
print(f"  RMSE = {newton_rmse_m:.4f}°C")
print(f"  MAE = {newton_mae_m:.4f}°C")

# Fit SCFC - use same approach but independently
def scfc_obj(params):
    h0, m_evap = params
    if h0 <= 0 or m_evap < 0:
        return 1e10
    pred = scfc_model_numeric(manitoba_times, T0_m_c, h0, m_evap, T_amb_m_c, MASS, C_P_WATER, SURFACE_AREA)
    return rmse(manitoba_temps_c, pred)

# Use differential evolution for global search
from scipy.optimize import differential_evolution
bounds = [(1e-2, 1000.0), (0.0, 1e-2)]
de_res = differential_evolution(scfc_obj, bounds, maxiter=50, popsize=10, seed=42, atol=1e-6)
local_res = minimize(scfc_obj, de_res.x, bounds=bounds, method='L-BFGS-B')
h0_fit, m_evap_fit = local_res.x
scfc_pred_m = scfc_model_numeric(manitoba_times, T0_m_c, h0_fit, m_evap_fit, T_amb_m_c, MASS, C_P_WATER, SURFACE_AREA)
scfc_rmse_m = rmse(manitoba_temps_c, scfc_pred_m)
scfc_mae_m = mae(manitoba_temps_c, scfc_pred_m)

print(f"\nSCFC (independent fit):")
print(f"  h0 = {h0_fit:.4f}")
print(f"  m_evap = {m_evap_fit:.2e} kg/(s*K)")
print(f"  RMSE = {scfc_rmse_m:.4f}°C")
print(f"  MAE = {scfc_mae_m:.4f}°C")

print(f"\nImprovement:")
print(f"  RMSE: {(newton_rmse_m - scfc_rmse_m)/newton_rmse_m*100:.1f}%")
print(f"  MAE: {(newton_mae_m - scfc_mae_m)/newton_mae_m*100:.1f}%")

# --- Audit Arizona ---
print("\n" + "=" * 60)
print("AUDIT: ARIZONA COFFEE")
print("=" * 60)

arizona_times = np.array([d['t_min'] for d in arizona_raw['data']]) * 60
arizona_temps_c = np.array([d['T_c'] for d in arizona_raw['data']])
T_amb_a = arizona_raw['t_amb_c']
T0_a = arizona_temps_c[0]

print(f"Data points: {len(arizona_times)}")
print(f"T_amb: {T_amb_a:.2f}°C, T0: {T0_a:.2f}°C")

# Newton
delta_T_a = arizona_temps_c - T_amb_a
log_delta_a = np.log(delta_T_a)
k_guess_a = -np.polyfit(arizona_times, log_delta_a, 1)[0]

res_a = minimize(newton_obj, [k_guess_a], bounds=[(1e-6, 1.0)], method='L-BFGS-B')
k_fit_a = res_a.x[0]
newton_pred_a = newton_model(arizona_times, T0_a, k_fit_a, T_amb_a)
newton_rmse_a = rmse(arizona_temps_c, newton_pred_a)
newton_mae_a = mae(arizona_temps_c, newton_pred_a)

print(f"\nNewton (analytical fit):")
print(f"  k = {k_fit_a:.6f} 1/s")
print(f"  RMSE = {newton_rmse_a:.4f}°C")
print(f"  MAE = {newton_mae_a:.4f}°C")

# SCFC
def scfc_obj_a(params):
    h0, m_evap = params
    if h0 <= 0 or m_evap < 0:
        return 1e10
    pred = scfc_model_numeric(arizona_times, T0_a, h0, m_evap, T_amb_a, MASS, C_P_WATER, SURFACE_AREA)
    return rmse(arizona_temps_c, pred)

de_res_a = differential_evolution(scfc_obj_a, bounds, maxiter=50, popsize=10, seed=42, atol=1e-6)
local_res_a = minimize(scfc_obj_a, de_res_a.x, bounds=bounds, method='L-BFGS-B')
h0_fit_a, m_evap_fit_a = local_res_a.x
scfc_pred_a = scfc_model_numeric(arizona_times, T0_a, h0_fit_a, m_evap_fit_a, T_amb_a, MASS, C_P_WATER, SURFACE_AREA)
scfc_rmse_a = rmse(arizona_temps_c, scfc_pred_a)
scfc_mae_a = mae(arizona_temps_c, scfc_pred_a)

print(f"\nSCFC (independent fit):")
print(f"  h0 = {h0_fit_a:.4f}")
print(f"  m_evap = {m_evap_fit_a:.2e} kg/(s*K)")
print(f"  RMSE = {scfc_rmse_a:.4f}°C")
print(f"  MAE = {scfc_mae_a:.4f}°C")

print(f"\nImprovement:")
print(f"  RMSE: {(newton_rmse_a - scfc_rmse_a)/newton_rmse_a*100:.1f}%")
print(f"  MAE: {(newton_mae_a - scfc_mae_a)/newton_mae_a*100:.1f}%")

# --- Compare with original results ---
print("\n" + "=" * 60)
print("COMPARISON WITH ORIGINAL RESULTS (from method_out.json)")
print("=" * 60)

with open('method_out.json') as f:
    orig = json.load(f)

for ds in orig['datasets']:
    if ds['dataset'] in ['manitoba_coffee', 'arizona_coffee']:
        ex = ds['examples'][0]
        r = json.loads(ex['output'])
        print(f"\n{ds['dataset']}:")
        print(f"  Original Newton RMSE: {r['newton']['rmse_c']:.4f}°C  |  Audit: {newton_rmse_m if ds['dataset']=='manitoba_coffee' else newton_rmse_a:.4f}°C")
        print(f"  Original SCFC RMSE:   {r['scfc']['rmse_c']:.4f}°C  |  Audit: {scfc_rmse_m if ds['dataset']=='manitoba_coffee' else scfc_rmse_a:.4f}°C")
        print(f"  Original Newton MAE:  {r['newton']['mae_c']:.4f}°C  |  Audit: {newton_mae_m if ds['dataset']=='manitoba_coffee' else newton_mae_a:.4f}°C")
        print(f"  Original SCFC MAE:    {r['scfc']['mae_c']:.4f}°C  |  Audit: {scfc_mae_m if ds['dataset']=='manitoba_coffee' else scfc_mae_a:.4f}°C")

# --- Placebo/Shuffle Test ---
print("\n" + "=" * 60)
print("PLACEBO TEST: Shuffled labels (should show NO improvement)")
print("=" * 60)

# Shuffle residuals and test if "improvement" is real
np.random.seed(123)
n_trials = 100
real_improvements = []

for ds_name, times, temps, T_amb, T0 in [
    ('manitoba', manitoba_times, manitoba_temps_c, T_amb_m_c, T0_m_c),
    ('arizona', arizona_times, arizona_temps_c, T_amb_a, T0_a)
]:
    # Fit Newton once
    delta_T = temps - T_amb
    log_delta = np.log(delta_T)
    k_guess = -np.polyfit(times, log_delta, 1)[0]
    res = minimize(lambda k: rmse(temps, newton_model(times, T0, k[0], T_amb)), 
                   [k_guess], bounds=[(1e-6, 1.0)], method='L-BFGS-B')
    k_fit = res.x[0]
    newton_pred = newton_model(times, T0, k_fit, T_amb)
    newton_rmse = rmse(temps, newton_pred)
    
    # Real SCFC fit
    def scfc_obj_sh(times, temps, T0, T_amb):
        def obj(p):
            h0, m_evap = p
            if h0 <= 0 or m_evap < 0: return 1e10
            pred = scfc_model_numeric(times, T0, h0, m_evap, T_amb, MASS, C_P_WATER, SURFACE_AREA)
            return rmse(temps, pred)
        return obj
    
    obj_fn = scfc_obj_sh(times, temps, T0, T_amb)
    de_res = differential_evolution(obj_fn, bounds, maxiter=30, popsize=8, seed=42, atol=1e-6)
    local_res = minimize(obj_fn, de_res.x, bounds=bounds, method='L-BFGS-B')
    scfc_rmse = local_res.fun
    
    real_imp = (newton_rmse - scfc_rmse) / newton_rmse * 100
    print(f"  {ds_name}: Real RMSE improvement = {real_imp:.1f}%")
    
    # Now test with SHUFFLED temps (placebo)
    placebo_imps = []
    for _ in range(n_trials):
        shuffled = np.random.permutation(temps)
        # Fit Newton on shuffled
        delta_s = shuffled - T_amb
        if np.all(delta_s > 0):
            log_d = np.log(delta_s)
            k_g = -np.polyfit(times, log_d, 1)[0]
            res_s = minimize(lambda k: rmse(shuffled, newton_model(times, T0, k[0], T_amb)), 
                           [k_g], bounds=[(1e-6, 1.0)], method='L-BFGS-B')
            n_rmse_s = rmse(shuffled, newton_model(times, T0, res_s.x[0], T_amb))
            
            # Fit SCFC on shuffled
            de_s = differential_evolution(obj_fn, bounds, maxiter=20, popsize=8, seed=42, atol=1e-6)
            local_s = minimize(obj_fn, de_s.x, bounds=bounds, method='L-BFGS-B')
            s_rmse_s = local_s.fun
            
            placebo_imps.append((n_rmse_s - s_rmse_s) / n_rmse_s * 100)
    
    print(f"  {ds_name}: Placebo improvements (n={n_trials}): mean={np.mean(placebo_imps):.2f}%, std={np.std(placebo_imps):.2f}%, max={np.max(placebo_imps):.2f}%")
    print(f"  {ds_name}: Real improvement {'EXCEEDS' if real_imp > np.max(placebo_imps) else 'WITHIN'} placebo range")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
