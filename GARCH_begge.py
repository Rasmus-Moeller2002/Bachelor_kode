import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.stats as stats
import statsmodels.api as sm
import os

# --- 1. SÆT MAPPE OP TIL GRAFER ---
script_mappe = os.path.dirname(os.path.abspath(__file__))
bachelor_mappe = os.path.dirname(script_mappe)
grafer_mappe = os.path.join(bachelor_mappe, "Grafer")
if not os.path.exists(grafer_mappe):
    os.makedirs(grafer_mappe)

# ---------------------------------------------------------
# 2. Hent og klargør data
# ---------------------------------------------------------
print("Henter data...")
data = pd.read_csv("spx_total_return.csv", index_col=0)
data.index = pd.to_datetime(data.index)
priser = data['SPXT INDEX']
returns = 100 * np.log(priser).diff().dropna()

# ---------------------------------------------------------
# 3. Definer parametre og Split
# ---------------------------------------------------------
konfidensniveau = 0.95
z_score = stats.norm.ppf(1 - konfidensniveau) 
test_dage = 504
split_index = len(returns) - test_dage

# ---------------------------------------------------------
# 4. RULLENDE GARCH ESTIMERING (Normal OG Student-t)
# ---------------------------------------------------------
forecast_mu_norm, forecast_sigma_norm = [], []
current_params_norm = None

forecast_mu_t, forecast_sigma_t, forecast_nu_t = [], [], []
current_params_t = None

print(f"\nStarter rullende estimering af BEGGE modeller for {test_dage} dage...")

for i in range(test_dage):
    current_train = returns.iloc[i : split_index + i]
    
    # --- MODEL 1: NORMALFORDELING ---
    model_norm = arch_model(current_train, mean='Constant', vol='Garch', p=1, q=1, dist='Normal')
    if i == 0 or i % 1 == 0:
        res_norm = model_norm.fit(disp='off', show_warning=False)
        current_params_norm = res_norm.params
    else:
        res_norm = model_norm.fix(current_params_norm)
        
    pred_norm = res_norm.forecast(horizon=1, reindex=False)
    forecast_mu_norm.append(pred_norm.mean.iloc[-1].values[0])
    forecast_sigma_norm.append(np.sqrt(pred_norm.variance.iloc[-1].values[0]))
    
    # --- MODEL 2: STUDENT-T FORDELING ---
    model_t = arch_model(current_train, mean='Constant', vol='Garch', p=1, q=1, dist='StudentsT')
    if i == 0 or i % 1 == 0:
        res_t = model_t.fit(disp='off', show_warning=False)
        current_params_t = res_t.params
    else:
        res_t = model_t.fix(current_params_t)
        
    pred_t = res_t.forecast(horizon=1, reindex=False)
    forecast_mu_t.append(pred_t.mean.iloc[-1].values[0])
    forecast_sigma_t.append(np.sqrt(pred_t.variance.iloc[-1].values[0]))
    forecast_nu_t.append(current_params_t['nu']) 
    
    if (i + 1) % 1 == 0:
        print(f"Færdig med {i + 1} ud af {test_dage} dage...")

# ---------------------------------------------------------
# 5. Udregn VaR for begge modeller
# ---------------------------------------------------------
mu_series_norm = pd.Series(forecast_mu_norm, index=returns.index[split_index:])
sigma_series_norm = pd.Series(forecast_sigma_norm, index=returns.index[split_index:])
garch_var_norm = -(mu_series_norm + z_score * sigma_series_norm)

mu_series_t = pd.Series(forecast_mu_t, index=returns.index[split_index:])
sigma_series_t = pd.Series(forecast_sigma_t, index=returns.index[split_index:])
nu_series_t = pd.Series(forecast_nu_t, index=returns.index[split_index:])

# Vi omregner sigma til skaleringsparameteren lambda for t-fordelingen
lambda_series_t = sigma_series_t * np.sqrt((nu_series_t - 2) / nu_series_t)
t_scores = stats.t.ppf(1 - konfidensniveau, df=nu_series_t)
garch_var_t = -(mu_series_t + lambda_series_t * t_scores)

# ---------------------------------------------------------
# 6. Backtest Funktion 
# ---------------------------------------------------------
def backtest_tests(hits, p, model_navn):
    print(f"\n{'='*50}")
    print(f" TEST AF GARCH VaR ({model_navn.upper()})")
    print(f"{'='*50}")
    T = len(hits)
    N = hits.sum()
    p_obs = N / T
    print(f"Faktiske overskridelser (N): {N} (Forventet: {T * p:.2f})")

    def safe_log(v): return np.log(v) if v > 0 else 0
    
    # 1. Kupiec POF Test (Unconditional Coverage)
    print("--- 1. Kupiec POF Test (Unconditional) ---")
    
    # Vi implementerer Jorions log-formel med edge-case håndtering
    LR_uc = -2 * ((T - N) * safe_log(1 - p) + N * safe_log(p) - 
                  (T - N) * safe_log(1 - p_obs) - N * safe_log(p_obs))
    
    p_val_uc = 1 - stats.chi2.cdf(LR_uc, df=1)
    
    print(f"Kupiec LR-Statistik (LR_uc): {LR_uc:.4f}")
    print(f"P-værdi: {p_val_uc:.4f}")
    if p_val_uc > 0.05:
       print("Konklusion: Modellen GODKENDES (Vi kan ikke afvise H0).\n")
    else:
       print("Konklusion: Modellen AFVISES (Vi afviser H0).\n")
    
    # 2. Christoffersen Test (Independence)
    print("\n--- 2. Christoffersen Independence Test ---")
    T00 = T01 = T10 = T11 = 0
    hits_array = hits.values
    
    # Optælling af overgange (T_ij)
    for i in range(1, len(hits_array)):
        if hits_array[i-1] == 0 and hits_array[i] == 0: T00 += 1
        elif hits_array[i-1] == 0 and hits_array[i] == 1: T01 += 1
        elif hits_array[i-1] == 1 and hits_array[i] == 0: T10 += 1
        elif hits_array[i-1] == 1 and hits_array[i] == 1: T11 += 1

    # Estimerede overgangssandsynligheder (pi_01 og pi_11 fra bogen)
    pi_01 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0
    pi_11 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0
    pi_00 = 1 - pi_01
    pi_10 = 1 - pi_11

    # Den samlede sandsynlighed under uafhængighed (pi_hat)
    pi_hat = (T01 + T11) / (T00 + T01 + T10 + T11)

    # Log-likelihood for uafhængighed (L(\hat{\Pi}))
    LL_indep = (T00 + T10) * safe_log(1 - pi_hat) + (T01 + T11) * safe_log(pi_hat)
    
    # Log-likelihood for første-ordens Markov afhængighed (L(\hat{\Pi}_1))
    LL_dep = T00 * safe_log(pi_00) + T01 * safe_log(pi_01) + T10 * safe_log(pi_10) + T11 * safe_log(pi_11)
    
    # Likelihood Ratio Statistik (LR_ind)
    LR_ind = -2 * (LL_indep - LL_dep)
    p_val_ind = 1 - stats.chi2.cdf(LR_ind, df=1)
    
    print(f"Christoffersen LR-Statistik (LR_ind): {LR_ind:.4f}")
    print(f"P-værdi: {p_val_ind:.4f}")
    if p_val_ind > 0.05:
       print("Konklusion: Modellen GODKENDES (Overskridelser er uafhængige).\n")
    else:
       print("Konklusion: Modellen AFVISES (Overskridelser klumper sammen).\n")
    
    # 3. Conditional Coverage (LR_cc)
    LR_cc = LR_uc + LR_ind
    p_val_cc = 1 - stats.chi2.cdf(LR_cc, df=2)
    print(f"Christoffersen CC LR-Statistik (LR_cc): {LR_cc:.4f}, P-værdi: {p_val_cc:.4f}")
    if p_val_cc > 0.05:
        print("-> Konklusion: Modellen GODKENDES i den kombinerede test.\n")
    else:
        print("-> Konklusion: Modellen AFVISES i den kombinerede test.\n")

test_returns = returns.iloc[split_index:]
hits_norm = (test_returns < -garch_var_norm).astype(int)
hits_t = (test_returns < -garch_var_t).astype(int)

backtest_tests(hits_norm, (1 - konfidensniveau), "GARCH-Normal")
backtest_tests(hits_t, (1 - konfidensniveau), "GARCH-Student-t")

# ---------------------------------------------------------
# 7. VISUALISERING 1: Kombineret VaR Plot
# ---------------------------------------------------------
breaches_norm = test_returns[hits_norm == 1]
breaches_t = test_returns[hits_t == 1]

plt.figure(figsize=(14, 7))
plt.plot(test_returns.index, test_returns, label="Faktisk Afkast (%)", color='steelblue', alpha=0.4)
plt.plot(garch_var_norm.index, -garch_var_norm, label="GARCH-Normal VaR (95%)", color='blue', linewidth=2)
plt.plot(garch_var_t.index, -garch_var_t, label="GARCH-t VaR (95%)", color='darkorange', linewidth=2, linestyle='--')

plt.scatter(breaches_norm.index, breaches_norm, facecolor='none', edgecolor='blue', s=100, linewidths=2, label=f"Overskridelser: Normal ({len(breaches_norm)} stk)", zorder=5)
plt.scatter(breaches_t.index, breaches_t, color='orange', edgecolor='darkorange', s=40, label=f"Overskridelser: t-fordeling ({len(breaches_t)} stk)", zorder=6)

plt.title("Backtest af Rullende GARCH(1,1) VaR: Normal vs. t-fordeling")
plt.ylabel("Dagligt log-afkast (%)")
plt.legend(loc='upper left', framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(grafer_mappe, "Backtest_GARCH_Kombineret_Synlig.png"), dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 8. VISUALISERING 2: Sammenlignende Histogram (Normal vs. t)
# ---------------------------------------------------------
# Vi bruger residualerne fra t-modellen som vores datagrundlag
std_resid_t = (test_returns - mu_series_t) / sigma_series_t

plt.figure(figsize=(10, 6))

# 1. Plot histogrammet af de faktiske residualer
plt.hist(std_resid_t, bins=45, density=True, alpha=0.5, color='steelblue', label='Faktiske Std. Residualer (Out-of-sample)')

# Sæt x-aksen op
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)

# 2. Plot Teoretisk Normalfordeling (Sort, solid)
pdf_norm = stats.norm.pdf(x, 0, 1)
plt.plot(x, pdf_norm, 'blue', linewidth=2, label='Normalfordeling')

# 3. Plot Teoretisk Student-t fordeling (Orange, stiplet)
# Vi bruger gennemsnittet af de rullende frihedsgrader
mean_nu = nu_series_t.mean()
skala = np.sqrt((mean_nu - 2) / mean_nu)
pdf_t = stats.t.pdf(x / skala, df=mean_nu) / skala 
plt.plot(x, pdf_t, color='darkorange', linestyle='--', linewidth=2.5, label=f't-fordeling (nu={mean_nu:.1f})')

plt.title("Tæthedsfordeling: Normal vs. Student-t")
plt.xlabel("Standardiserede Residualer")
plt.ylabel("Tæthed (Density)")
plt.legend(loc='upper left', framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(grafer_mappe, "Sammenligning_PDF_Normal_vs_t.png"), dpi=300, bbox_inches='tight')
plt.show()
