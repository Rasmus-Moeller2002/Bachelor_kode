import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate # TILFØJET TIL INTEGRATION
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import os

# --- 1. SÆT MAPPE OP TIL GRAFER ---
script_mappe = os.path.dirname(os.path.abspath(__file__))
bachelor_mappe = os.path.dirname(script_mappe)
grafer_mappe = os.path.join(bachelor_mappe, "Grafer")
if not os.path.exists(grafer_mappe):
    os.makedirs(grafer_mappe)

# --- 2. HENT DATA ---
df = pd.read_csv("spx_total_return.csv", index_col=0)
priser = df['SPXT INDEX']
afkast = 100 * np.log(priser).diff().dropna() 

# --- DEFINER KONFIDENSNIVEAUER ---
alpha_95 = 1 - 0.95
alpha_99 = 1 - 0.99

z_95 = stats.norm.ppf(alpha_95)
z_99 = stats.norm.ppf(alpha_99)

# ------------------------------------------
# A) GARCH Normal
# ------------------------------------------
am_norm = arch_model(afkast, mean='Constant', vol='Garch', p=1, q=1, dist='Normal')
res_norm = am_norm.fit(disp='off')
forecast_norm = res_norm.forecast(horizon=1)

mu_garch_norm = forecast_norm.mean.iloc[-1].values[0]
sigma_garch_norm = np.sqrt(forecast_norm.variance.iloc[-1].values[0])

# ------------------------------------------
# B) GARCH Student-t
# ------------------------------------------
am_t = arch_model(afkast, mean='Constant', vol='Garch', p=1, q=1, dist='StudentsT')
res_t = am_t.fit(disp='off')
forecast_t = res_t.forecast(horizon=1)

mu_garch_t = forecast_t.mean.iloc[-1].values[0]
sigma_garch_t = np.sqrt(forecast_t.variance.iloc[-1].values[0])
df_garch = res_t.params['nu']

# ==========================================
# 3. MONTE CARLO SIMULERING (10-DAGES VaR)
# ==========================================
np.random.seed(42) #42
N_sim = 10000      
horisont = 10      

print(f"\nStarter Monte Carlo Simulering ({N_sim} stier, {horisont} dage)...")

# ------------------------------------------
# A) Simulering for GARCH Normal
# ------------------------------------------
mu_n    = res_norm.params['mu']
omega_n = res_norm.params['omega']
alpha_n = res_norm.params['alpha[1]']
beta_n  = res_norm.params['beta[1]']

sim_var_norm    = np.zeros((N_sim, horisont))
sim_stød_norm   = np.zeros((N_sim, horisont)) 
sim_afkast_norm = np.zeros((N_sim, horisont)) 

epsilon_norm = np.random.standard_normal((N_sim, horisont))

sim_var_norm[:, 0] = forecast_norm.variance.iloc[-1].values[0]
sim_stød_norm[:, 0] = np.sqrt(sim_var_norm[:, 0]) * epsilon_norm[:, 0]
sim_afkast_norm[:, 0] = mu_n + sim_stød_norm[:, 0]

for t in range(1, horisont):
    sim_var_norm[:, t] = omega_n + alpha_n * (sim_stød_norm[:, t-1]**2) + beta_n * sim_var_norm[:, t-1]
    sim_stød_norm[:, t] = np.sqrt(sim_var_norm[:, t]) * epsilon_norm[:, t]
    sim_afkast_norm[:, t] = mu_n + sim_stød_norm[:, t]

# ÆNDRING: Transformer summen af log-afkast til simple procentvise afkast
kumulativt_log_afkast_norm = np.sum(sim_afkast_norm, axis=1)
kumulativt_afkast_norm = (np.exp(kumulativt_log_afkast_norm / 100) - 1) * 100

mc_var_norm_95 = np.percentile(kumulativt_afkast_norm, alpha_95 * 100)
mc_var_norm_99 = np.percentile(kumulativt_afkast_norm, alpha_99 * 100)
es_garch_norm_95 = np.mean(kumulativt_afkast_norm[kumulativt_afkast_norm < mc_var_norm_95])
es_garch_norm_99 = np.mean(kumulativt_afkast_norm[kumulativt_afkast_norm < mc_var_norm_99])


# ------------------------------------------
# B) Simulering for GARCH Student-t
# ------------------------------------------
mu_t    = res_t.params['mu']
omega_t = res_t.params['omega']
alpha_t = res_t.params['alpha[1]']
beta_t  = res_t.params['beta[1]']

sim_var_t    = np.zeros((N_sim, horisont))
sim_stød_t   = np.zeros((N_sim, horisont)) 
sim_afkast_t = np.zeros((N_sim, horisont))

stød_rå_t = np.random.standard_t(df_garch, size=(N_sim, horisont))
skaleringsfaktor_mc = np.sqrt((df_garch - 2) / df_garch) 
epsilon_t = stød_rå_t * skaleringsfaktor_mc

sim_var_t[:, 0] = forecast_t.variance.iloc[-1].values[0]
sim_stød_t[:, 0] = np.sqrt(sim_var_t[:, 0]) * epsilon_t[:, 0]
sim_afkast_t[:, 0] = mu_t + sim_stød_t[:, 0]

for t in range(1, horisont):
    sim_var_t[:, t] = omega_t + alpha_t * (sim_stød_t[:, t-1]**2) + beta_t * sim_var_t[:, t-1]
    sim_stød_t[:, t] = np.sqrt(sim_var_t[:, t]) * epsilon_t[:, t]
    sim_afkast_t[:, t] = mu_t + sim_stød_t[:, t]

# ÆNDRING: Transformer summen af log-afkast til simple procentvise afkast
kumulativt_log_afkast_t = np.sum(sim_afkast_t, axis=1)
kumulativt_afkast_t = (np.exp(kumulativt_log_afkast_t / 100) - 1) * 100

mc_var_t_95 = np.percentile(kumulativt_afkast_t, alpha_95 * 100)
mc_var_t_99 = np.percentile(kumulativt_afkast_t, alpha_99 * 100)
es_garch_t_95 = np.mean(kumulativt_afkast_t[kumulativt_afkast_t < mc_var_t_95])
es_garch_t_99 = np.mean(kumulativt_afkast_t[kumulativt_afkast_t < mc_var_t_99])


# ------------------------------------------
# C) Simulering & Analytisk for Statisk 
# ------------------------------------------
mu_statisk = np.mean(afkast)
sigma_statisk = np.std(afkast)
df_statisk = stats.t.fit(afkast)[0]
lambda_scale = sigma_statisk * np.sqrt((df_statisk - 2) / df_statisk)

# --- 1. Analytisk 10-dags Normalfordeling via FORMEL 2.2 ---
# Først skal vi bruge decimaler til log-normal integralet
mu_1_dec = mu_statisk / 100
sigma_1_dec = sigma_statisk / 100
mu_10_dec = mu_1_dec * horisont
sigma_10_dec = sigma_1_dec * np.sqrt(horisont)

def lognormal_pdf(x, mu, sigma):
    if x <= 0: return 0.0
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))

# Analytisk VaR 
q_95 = np.exp(mu_10_dec + z_95 * sigma_10_dec) 
q_99 = np.exp(mu_10_dec + z_99 * sigma_10_dec) 

analytisk_var_norm_95 = (q_95 - 1) * 100
analytisk_var_norm_99 = (q_99 - 1) * 100

# Analytisk ES via integral (Formel 2.2)
taeller_95, _ = integrate.quad(lambda x: x * lognormal_pdf(x, mu_10_dec, sigma_10_dec), 0, q_95)
naevner_95, _ = integrate.quad(lambda x: lognormal_pdf(x, mu_10_dec, sigma_10_dec), 0, q_95)
analytisk_es_norm_95 = ((taeller_95 / naevner_95) - 1) * 100

taeller_99, _ = integrate.quad(lambda x: x * lognormal_pdf(x, mu_10_dec, sigma_10_dec), 0, q_99)
naevner_99, _ = integrate.quad(lambda x: lognormal_pdf(x, mu_10_dec, sigma_10_dec), 0, q_99)
analytisk_es_norm_99 = ((taeller_99 / naevner_99) - 1) * 100


# --- 2. Statisk Normalfordeling (MC Simulering) ---
sim_afkast_statisk_norm = mu_statisk + sigma_statisk * np.random.standard_normal((N_sim, horisont))

# ÆNDRING: Transformer summen af log-afkast til simple procentvise afkast
kumulativt_log_statisk_norm = np.sum(sim_afkast_statisk_norm, axis=1)
kumulativt_afkast_statisk_norm = (np.exp(kumulativt_log_statisk_norm / 100) - 1) * 100

mc_var_statisk_norm_95 = np.percentile(kumulativt_afkast_statisk_norm, alpha_95 * 100)
mc_var_statisk_norm_99 = np.percentile(kumulativt_afkast_statisk_norm, alpha_99 * 100)
es_statisk_norm_95 = np.mean(kumulativt_afkast_statisk_norm[kumulativt_afkast_statisk_norm < mc_var_statisk_norm_95])
es_statisk_norm_99 = np.mean(kumulativt_afkast_statisk_norm[kumulativt_afkast_statisk_norm < mc_var_statisk_norm_99])


# --- 3. Statisk Student-t fordeling (MC Simulering) ---
stød_rå_statisk_t = np.random.standard_t(df_statisk, size=(N_sim, horisont))
sim_afkast_statisk_t = mu_statisk + lambda_scale * stød_rå_statisk_t

# ÆNDRING: Transformer summen af log-afkast til simple procentvise afkast
kumulativt_log_statisk_t = np.sum(sim_afkast_statisk_t, axis=1)
kumulativt_afkast_statisk_t = (np.exp(kumulativt_log_statisk_t / 100) - 1) * 100

mc_var_statisk_t_95 = np.percentile(kumulativt_afkast_statisk_t, alpha_95 * 100)
mc_var_statisk_t_99 = np.percentile(kumulativt_afkast_statisk_t, alpha_99 * 100)
es_statisk_t_95 = np.mean(kumulativt_afkast_statisk_t[kumulativt_afkast_statisk_t < mc_var_statisk_t_95])
es_statisk_t_99 = np.mean(kumulativt_afkast_statisk_t[kumulativt_afkast_statisk_t < mc_var_statisk_t_99])


# ==========================================
# 5. PRINT ENDELIGE 10-DAGES RESULTATER
# ==========================================
print("\n" + "="*70)
print(f" ENDELIGE 10-DAGES RISIKOMÅL (Simple Afkast)")
print("="*70)

print("--- STATISK NORMAL (Analytisk 'Facit' via Formel 2.2) ---")
print(f"95% Konfidens: VaR = {analytisk_var_norm_95:.4f} %  |  ES = {analytisk_es_norm_95:.4f} %")
print(f"99% Konfidens: VaR = {analytisk_var_norm_99:.4f} %  |  ES = {analytisk_es_norm_99:.4f} %")

print("\n--- STATISKE MODELLER (IID Monte Carlo) ---")
print(f"Normal (95%):  VaR = {mc_var_statisk_norm_95:.4f} %  |  ES = {es_statisk_norm_95:.4f} %")
print(f"Normal (99%):  VaR = {mc_var_statisk_norm_99:.4f} %  |  ES = {es_statisk_norm_99:.4f} %")
print(f"Stud-t (95%):  VaR = {mc_var_statisk_t_95:.4f} %  |  ES = {es_statisk_t_95:.4f} %")
print(f"Stud-t (99%):  VaR = {mc_var_statisk_t_99:.4f} %  |  ES = {es_statisk_t_99:.4f} %")

print("\n--- DYNAMISKE MODELLER (GARCH Monte Carlo) ---")
print(f"G-Norm (95%):  VaR = {mc_var_norm_95:.4f} %  |  ES = {es_garch_norm_95:.4f} %")
print(f"G-Norm (99%):  VaR = {mc_var_norm_99:.4f} %  |  ES = {es_garch_norm_99:.4f} %")
print(f"G-Stud (95%):  VaR = {mc_var_t_95:.4f} %  |  ES = {es_garch_t_95:.4f} %")
print(f"G-Stud (99%):  VaR = {mc_var_t_99:.4f} %  |  ES = {es_garch_t_99:.4f} %")
print("="*70)

# ==========================================
# 6. HISTORISK SIMULATION (10-DAGES VaR)
# ==========================================
# ÆNDRING: Omregn de historiske 10-dages sum af log-afkast til simple afkast
afkast_10d_log = afkast.rolling(window=10).sum().dropna()
afkast_10d = (np.exp(afkast_10d_log / 100) - 1) * 100

# --- 95% Konfidensniveau ---
hs_var_95 = np.percentile(afkast_10d, alpha_95 * 100)
hs_es_95 = np.mean(afkast_10d[afkast_10d < hs_var_95])

# --- 99% Konfidensniveau ---
hs_var_99 = np.percentile(afkast_10d, alpha_99 * 100)
hs_es_99 = np.mean(afkast_10d[afkast_10d < hs_var_99])

print("\n--- HISTORISK SIMULATION (10-DAGE) ---")
print(f"Historisk (95%): VaR = {hs_var_95:.4f} %  |  ES = {hs_es_95:.4f} %")
print(f"Historisk (99%): VaR = {hs_var_99:.4f} %  |  ES = {hs_es_99:.4f} %")