#!/usr/bin/env python3
"""Quick audit of Arizona results only."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

with open('data_arizona.json') as f:
    arizona_raw = json.load(f)

L_V = 2260000
C_P_WATER = 4186
RHO_WATER = 1000
VOLUME = 0.3
MASS = VOLUME * RHO_WATER
RADIUS = 0.04
HEIGHT = 0.1
SURFACE_AREA = np.pi * RADIUS**2 + 2 * np.pi * RADIUS * HEIGHT

arizona_times = np.array([d['t_min'] for d in arizona_raw['data']]) * 60
arizona_temps_c = np.array([d['T_c'] for d in arizona_raw['data']])
T_amb_a = arizona_raw['t_amb_c']
T0_a = arizona_temps_c[0]

def newton_model(t, T0, k, T_amb):
    return T_amb + (T0 - T_amb) * np.exp(-k * t)

def scfc_deriv(t, T, h0, m_evap, T_amb, m, cp, A):
    dT = T - T_amb
    if dT <= 0: return 0.0
    Q_conv = h0 * A * (dT ** (4/3))
    Q_evap = m_evap * dT * L_V
    return -(Q_conv + Q_evap) / (m * cp)

def scfc_numeric(t_eval, T0, h0, m_evap, T_amb, m, cp, A):
    sol = solve_ivp(scfc_deriv, [t_eval[0], t_eval[-1]], [T0],
                    args=(h0, m_evap, T_amb, m, cp, A),
                    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.y[0]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Newton - use same approach as main (minimize with ODE)
def newton_obj(k):
    pred = newton_model(arizona_times, T0_a, k[0], T_amb_a)
    return rmse(arizona_temps_c, pred)

# Initial guess from log-linear
delta_T = arizona_temps_c - T_amb_a
log_delta = np.log(delta_T)
k_guess = -np.polyfit(arizona_times, log_delta, 1)[0]

res = minimize(newton_obj, [k_guess], bounds=[(1e-6, 1.0)], method='L-BFGS-B')
k_fit = res.x[0]
newton_pred = newton_model(arizona_times, T0_a, k_fit, T_amb_a)
n_rmse = rmse(arizona_temps_c, newton_pred)
n_mae = mae(arizona_temps_c, newton_pred)

print(f"Newton: k={k_fit:.6f}, RMSE={n_rmse:.4f}, MAE={n_mae:.4f}")

# SCFC
def scfc_obj(p):
    h0, m_evap = p
    if h0 <= 0 or m_evap < 0: return 1e10
    pred = scfc_numeric(arizona_times, T0_a, h0, m_evap, T_amb_a, MASS, C_P_WATER, SURFACE_AREA)
    return rmse(arizona_temps_c, pred)

from scipy.optimize import differential_evolution
bounds = [(1e-2, 1000.0), (0.0, 1e-2)]
de_res = differential_evolution(scfc_obj, bounds, maxiter=50, popsize=10, seed=42, atol=1e-6)
local_res = minimize(scfc_obj, de_res.x, bounds=bounds, method='L-BFGS-B')
h0_fit, m_evap_fit = local_res.x
scfc_pred = scfc_numeric(arizona_times, T0_a, h0_fit, m_evap_fit, T_amb_a, MASS, C_P_WATER, SURFACE_AREA)
s_rmse = rmse(arizona_temps_c, scfc_pred)
s_mae = mae(arizona_temps_c, scfc_pred)

print(f"SCFC: h0={h0_fit:.4f}, m_evap={m_evap_fit:.2e}, RMSE={s_rmse:.4f}, MAE={s_mae:.4f}")
print(f"Improvement: RMSE={(n_rmse-s_rmse)/n_rmse*100:.1f}%, MAE={(n_mae-s_mae)/n_mae*100:.1f}%")

# Compare with original
with open('method_out.json') as f:
    orig = json.load(f)
for ds in orig['datasets']:
    if ds['dataset'] == 'arizona_coffee':
        r = json.loads(ds['examples'][0]['output'])
        print(f"\nOriginal: Newton RMSE={r['newton']['rmse_c']:.4f}, SCFC RMSE={r['scfc']['rmse_c']:.4f}")
        print(f"Match: Newton RMSE diff={abs(r['newton']['rmse_c']-n_rmse):.6f}, SCFC RMSE diff={abs(r['scfc']['rmse_c']-s_rmse):.6f}")
