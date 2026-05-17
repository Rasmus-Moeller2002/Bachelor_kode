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

# Indlæs data
print("Henter data...")
data = pd.read_csv("spx_total_return.csv", index_col=0)
data.index = pd.to_datetime(data.index)
priser = data['SPXT INDEX']
afkast = 100 * np.log(priser).diff().dropna()    

konfidensniveau = 0.95
p_hale = 1 - konfidensniveau 
test_dage = 504
window_size = len(afkast) - test_dage

# --------------------
# 2. Backtest Funktion 
# --------------------
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

print(f"\nTotal antal dage i datasæt: {len(afkast)}")
print(f"Rullende vinduesstørrelse: {window_size} dage")
print(f"Testdage (Out-of-sample): {test_dage}")

# =========================================
# 4. Beregn VaR og ES for det FULDE datasæt
# =========================================
var_full = np.quantile(afkast, p_hale)
es_full = afkast[afkast <= var_full].mean()

# ====================
# 5. Rullende Backtest 
# ====================
rolling_var_hs = afkast.rolling(window=window_size).quantile(p_hale).shift(1)

test_returns = afkast.iloc[-test_dage:]
test_var_hs = rolling_var_hs.iloc[-test_dage:]

# Udfør VaR Tests
hits_hs = (test_returns < test_var_hs).astype(int)
backtest_tests(hits_hs, p_hale, "Historisk Simulation")

# --------------------------------
# 6. Print Endelige VaR Resultater
# --------------------------------
print(f"\n--- HISTORISK SIMULATION FOR I MORGEN ({konfidensniveau*100}%) ---")
print(f"Historisk VaR: {var_full:.4f}%")
print(f"Historisk ES: {es_full:.4f}%")

# ================
# 7. Visualisering
# ================
plt.figure(figsize=(12, 6))
plt.plot(test_returns.index, test_returns, label="Faktisk Dagligt Afkast (Testsæt)", color='steelblue', alpha=0.5)
plt.plot(test_var_hs.index, test_var_hs, color='purple', linestyle='-', linewidth=2, label=f"Historisk VaR ({int(konfidensniveau*100)}%)")

breaches_hs = test_returns[test_returns < test_var_hs]
antal_breaches_hs = len(breaches_hs)
plt.scatter(breaches_hs.index, breaches_hs, color='purple', s=60, zorder=5, label=f"Overskridelser: HS ({antal_breaches_hs} stk)")

plt.title("Backtest af Historisk Simulation VaR (Rullende Vindue)")
plt.xlabel("Dato")
plt.ylabel("Dagligt Log-afkast (%)")
plt.ylim(test_returns.min() - 0.025, test_returns.max() + 0.01)
plt.legend(loc='upper left', framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.tight_layout()

gem_sti = os.path.join(grafer_mappe, "Backtest_Historisk_VaR.png")
plt.savefig(gem_sti, dpi=300, bbox_inches='tight')
plt.show()