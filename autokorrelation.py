import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

data =  pd.read_csv("spx_total_return.csv", index_col=0)
priser = data['SPXT INDEX']
# Antager at du allerede har udregnet dine log_returns:
log_returns = 100 * np.log(priser).diff().dropna()

# --- NYT: Beregn kvadrerede markedsstød (ARCH-effekter) præcis som i R ---
mu = log_returns.mean()
at = log_returns - mu
at_sq = at**2

# ---------------------------------------------------------
# 1. Visuel Test: ACF Plot for Log-afkast
# ---------------------------------------------------------
# Vi tegner plottet, zero=False fjerner lag 0 (som altid er 1)
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(log_returns, lags=20, zero=False, ax=ax, alpha=0.05)

plt.title("Autocorrelation Function (ACF) for Log-afkast")
plt.xlabel("Lags (Dage)")
plt.ylabel("Autokorrelation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.show()

# ---------------------------------------------------------
# 2. Statistisk Test: Ljung-Box Test for Log-afkast
# ---------------------------------------------------------
print("\n--- Ljung-Box Test for Autokorrelation (Log-afkast) ---")
lb_test = acorr_ljungbox(log_returns, lags=[5, 10, 20], return_df=True)

lb_test.columns = ['Test-Statistik (Q)', 'P-værdi']
lb_test.index.name = 'Lags'
lb_test['P-værdi'] = lb_test['P-værdi'].apply(lambda x: '< 2.2e-16' if x < 2.2e-16 else f"{x:.6e}")
print(lb_test)

print("\nFortolkning:")
print("H0: Der er INGEN autokorrelation i dataen (Afkast er uafhængige).")
print("H1: Der ER autokorrelation i dataen.")


# ---------------------------------------------------------
# 3. Visuel Test: ACF Plot for Kvadrerede Stød (ARCH-effekter)
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
plot_acf(at_sq, lags=20, zero=False, ax=ax2, alpha=0.05, color='darkred')

plt.title("Autocorrelation Function (ACF) for Kvadrerede Markedsstød")
plt.xlabel("Lags (Dage)")
plt.ylabel("Autokorrelation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.show()

# ---------------------------------------------------------
# 4. Statistisk Test: Ljung-Box Test for Kvadrerede Stød
# ---------------------------------------------------------
print("\n--- Ljung-Box Test for ARCH-effekter (Kvadrerede stød) ---")
lb_test_sq = acorr_ljungbox(at_sq, lags=[5, 10, 20], return_df=True)

# Omdøb kolonnerne så de er på dansk og nemme at læse
lb_test_sq.columns = ['Test-Statistik (Q)', 'P-værdi']
lb_test_sq.index.name = 'Lags'
lb_test_sq['P-værdi'] = lb_test_sq['P-værdi'].apply(lambda x: '< 2.2e-16' if x < 2.2e-16 else f"{x:.6e}")
print(lb_test_sq)

# En lille hjælpende tekst til fortolkningen af kvadrerede stød:
print("\nFortolkning (ARCH-effekter):")
print("H0: Der er INGEN autokorrelation i de kvadrerede stød (Ingen ARCH-effekter).")
print("H1: Der ER autokorrelation (Volatilitetsklynger til stede - GARCH-modellen er retfærdiggjort!).")