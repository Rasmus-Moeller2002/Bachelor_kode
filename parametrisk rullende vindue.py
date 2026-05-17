import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# --- 1. SÆT MAPPE OP TIL GRAFER ---
script_mappe = os.path.dirname(os.path.abspath(__file__))
bachelor_mappe = os.path.dirname(script_mappe)
grafer_mappe = os.path.join(bachelor_mappe, "Grafer")
if not os.path.exists(grafer_mappe):
    os.makedirs(grafer_mappe)

#Indlæs data
print("Henter data...")
data = pd.read_csv("spx_total_return.csv", index_col=0)
data.index = pd.to_datetime(data.index)
priser = data['SPXT INDEX']
returns = 100*np.log(priser).diff().dropna()

# ----------------------------------------
# 2. Definer parametre og vinduesstørrelse
# ----------------------------------------
konfidensniveau = 0.95
test_dage = 504
window_size = len(returns) - test_dage

print(f"Total antal dage i datasæt: {len(returns)}")
print(f"Rullende vinduesstørrelse: {window_size} dage")
print(f"Testdage (Out-of-sample): {test_dage}")

# -------------------------
# DEFINER BACKTEST FUNKTION 
# -------------------------
def backtest_tests(hits, p, model_navn):
    print(f"\n{'='*50}")
    print(f" TEST AF PARAMETRISK VaR ({model_navn.upper()}) 5%")
    print(f"{'='*50}")
    
    T = len(hits)     # Totale antal observationer
    N = hits.sum()    # Antal overskridelser 
    p_obs = N / T     # Den observerede ratio (N/T)
    
    print(f"Observationer (T): {T}")
    print(f"Forventede overskridelser (T*p): {T * p:.2f}")
    print(f"Faktiske overskridelser (N): {N}")
    print(f"Faktisk overskridelses-ratio (N/T): {p_obs:.4f} (Forventet p: {p:.4f})\n")

    def safe_log(v):
        return np.log(v) if v > 0 else 0

    # 1. Kupiec POF Test 
    print("--- 1. Kupiec POF Test (Unconditional) ---")
    
    # Jorion formel
    LR_uc = -2 * ((T - N) * safe_log(1 - p) + N * safe_log(p) - 
                  (T - N) * safe_log(1 - p_obs) - N * safe_log(p_obs))
    
    p_val_uc = 1 - stats.chi2.cdf(LR_uc, df=1)
    
    print(f"Kupiec LR-Statistik (LR_uc): {LR_uc:.4f}")
    print(f"P-værdi: {p_val_uc:.4f}")
    if p_val_uc > 0.05:
       print("Konklusion: Modellen GODKENDES (Vi kan ikke afvise H0).\n")
    else:
       print("Konklusion: Modellen AFVISES (Vi afviser H0).\n")

    # 2. Christoffersen Test 
    print("\n--- 2. Christoffersen Independence Test ---")
    N00 = N01 = N10 = N11 = 0
    hits_array = hits.values
    
    # Optælling af overgange (T_ij)
    for i in range(1, len(hits_array)):
        if hits_array[i-1] == 0 and hits_array[i] == 0: N00 += 1
        elif hits_array[i-1] == 0 and hits_array[i] == 1: N01 += 1
        elif hits_array[i-1] == 1 and hits_array[i] == 0: N10 += 1
        elif hits_array[i-1] == 1 and hits_array[i] == 1: N11 += 1

    # Estimerede overgangssandsynligheder 
    pi_01 = N01 / (N00 + N01) if (N00 + N01) > 0 else 0
    pi_11 = N11 / (N10 + N11) if (N10 + N11) > 0 else 0
    pi_00 = 1 - pi_01
    pi_10 = 1 - pi_11

    # Den samlede sandsynlighed under uafhængighed 
    pi_hat = (N01 + N11) / (N00 + N01 + N10 + N11)

    # Log-likelihood for uafhængighed 
    LL_indep = (N00 + N10) * safe_log(1 - pi_hat) + (N01 + N11) * safe_log(pi_hat)
    
    # Log-likelihood for første-ordens Markov afhængighed 
    LL_dep = N00 * safe_log(pi_00) + N01 * safe_log(pi_01) + N10 * safe_log(pi_10) + N11 * safe_log(pi_11)
    
    # Likelihood Ratio Statistik 
    LR_ind = -2 * (LL_indep - LL_dep)
    p_val_ind = 1 - stats.chi2.cdf(LR_ind, df=1)
    
    print(f"Christoffersen LR-Statistik (LR_ind): {LR_ind:.4f}")
    print(f"P-værdi: {p_val_ind:.4f}")
    if p_val_ind > 0.05:
       print("Konklusion: Modellen GODKENDES (Overskridelser er uafhængige).\n")
    else:
       print("Konklusion: Modellen AFVISES (Overskridelser klumper sammen).\n")

    # 3. Christoffersen Conditional Coverage 
    print("\n--- 3. Christoffersen Conditional Coverage Test ---")
    LR_cc = LR_uc + LR_ind
    p_val_cc = 1 - stats.chi2.cdf(LR_cc, df=2)
    
    print(f"Christoffersen CC LR-Statistik (LR_cc): {LR_cc:.4f}")
    print(f"P-værdi: {p_val_cc:.4f}")
    if p_val_cc > 0.05:
        print("-> Konklusion: Modellen GODKENDES i den kombinerede test.\n")
    else:
        print("-> Konklusion: Modellen AFVISES i den kombinerede test.\n")        

# ----------------------------------------------------------
# 3. Træn den RULLENDE Statiske Model (Normal + t-fordeling)
# ----------------------------------------------------------
rolling_mu = returns.rolling(window=window_size).mean().shift(1)
rolling_sigma = returns.rolling(window=window_size).std().shift(1)

# A) NORMAL VaR
z_score = stats.norm.ppf(1 - konfidensniveau)
rolling_var_norm = (rolling_mu + z_score * rolling_sigma)

# B) T-FORDELING VaR
train_returns = returns.iloc[:window_size]
df_t, loc_t, scale_t = stats.t.fit(train_returns)
print(f"Estimerede frihedsgrader (df) for t-fordelingen: {df_t:.2f}")

# Skalering, side 95 i statistikbogen
lambda_scale = rolling_sigma * np.sqrt((df_t - 2) / df_t)
t_score = stats.t.ppf(1 - konfidensniveau, df=df_t)
rolling_var_t = (rolling_mu + lambda_scale * t_score)

# ----------------------------------------
# 4. Klargør Testdata (De sidste 504 dage)
# ----------------------------------------
test_returns = returns.iloc[-test_dage:]
test_var_norm = rolling_var_norm.iloc[-test_dage:]
test_var_t = rolling_var_t.iloc[-test_dage:]

# --------------------------------
# 5. Visualisering af resultaterne
# --------------------------------
plt.figure(figsize=(12, 6))

plt.plot(test_returns.index, test_returns, label="Faktisk Dagligt log-afkast (Testsæt)", color='steelblue', alpha=0.5)

# Plot Normal VaR
plt.plot(test_var_norm.index, test_var_norm, color='olivedrab', linestyle='--', linewidth=2, label=f"Statisk-Normal VaR ({int(konfidensniveau*100)}%)")

# Plot T-fordeling VaR
plt.plot(test_var_t.index, test_var_t, color='purple', linestyle='--', linewidth=2, label=f"Statisk-t VaR ({int(konfidensniveau*100)}%)")

# Find overskridelser
breaches_norm = test_returns[test_returns < test_var_norm]
breaches_t = test_returns[test_returns < test_var_t]

antal_breaches_norm = len(breaches_norm)
antal_breaches_t = len(breaches_t)

# 1. Plot Normalfordelingens overskridelser FØRST (store blå prikker)
plt.scatter(breaches_norm.index, breaches_norm, facecolor='none', edgecolor='olivedrab', s=80, linewidths=2, zorder=5, 
            label=f"Overskridelser: Statisk-Normal ({antal_breaches_norm} stk)")

# 2. Plot t-fordelingens overskridelser OVENPÅ (mindre orange prikker)
plt.scatter(breaches_t.index, breaches_t, color='purple', s=25, zorder=6, 
            label=f"Overskridelser: Statisk-t ({antal_breaches_t} stk)")

plt.title("Backtest af Rullende Statisk VaR (Normalfordeling vs. t-fordeling)")
plt.xlabel("Dato")
plt.ylabel("Dagligt Log-afkast")

#Skaber plads i bunden af plottet
plt.ylim(test_returns.min() - 1, test_returns.max() + 1)

# plot backtestresultaterne
plt.legend(loc='upper left', framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.tight_layout()
gem_sti = os.path.join(grafer_mappe, "Backtest_parametrisk.png")
plt.savefig(gem_sti, dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# 6. Backtest Evaluering
# ----------------------
p_expected = 1 - konfidensniveau

# Udregn overskridelser for begge modeller
hits_norm = (test_returns < test_var_norm).astype(int) 
hits_t = (test_returns < test_var_t).astype(int) 

# Kør backtest for Normalfordelingen
backtest_tests(hits_norm, p_expected, "Normal")

# Kør backtest for T-fordelingen
backtest_tests(hits_t, p_expected, "Student-T")
