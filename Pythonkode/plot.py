import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# --- 1. SÆT MAPPE OP TIL GRAFER ---
script_mappe = os.path.dirname(os.path.abspath(__file__))
bachelor_mappe = os.path.dirname(script_mappe)
grafer_mappe = os.path.join(bachelor_mappe, "Grafer")
if not os.path.exists(grafer_mappe):
    os.makedirs(grafer_mappe)

# --- 2. HENT DATA ---
# Vi bruger HELE datasættet til det endelige forecast
df = pd.read_csv("spx_total_return.csv", index_col=0)
priser = df['SPXT INDEX']
# Ganger med 100 som aftalt for GARCH-optimeringen
afkast = 100 * np.log(priser).diff().dropna()

# ==========================================
# 1. Udregn parametre til fordelingerne
# ==========================================
# Parametre for Normalfordeling
mu_norm = np.mean(afkast)
sigma_norm = np.std(afkast, ddof=1) # ddof=1 for stikprøve-standardafvigelse

# Parametre for Student-t fordeling (Fitter data for at finde de bedste parametre)
df_t, loc_t, scale_t = stats.t.fit(afkast)


# ==========================================
# PLOT 1: QQ-Plots Side-om-Side (HELT RENE)
# ==========================================
fig2, (ax_qq1, ax_qq2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# --- QQ-Plot 1: Normalfordeling ---
stats.probplot(afkast, dist="norm", plot=ax_qq1)

# Sørger for at der ikke er uønsket tekst, og sætter vores egne titler
ax_qq1.set_title("QQ-Plot: Normalfordeling", fontweight='bold', fontsize=14)
ax_qq1.set_xlabel("Teoretiske Kvantiler", fontsize=12)
ax_qq1.set_ylabel(" Empiriske Kvantiler", fontsize=12)

# Formatering af prikker og streg
prikker_norm, streg_norm = ax_qq1.get_lines()
prikker_norm.set_markerfacecolor('black')
prikker_norm.set_markeredgecolor('black') # Sort kant giver et skarpere og renere look
prikker_norm.set_alpha(0.5)
streg_norm.set_color('red')
streg_norm.set_linewidth(2)

ax_qq1.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

# --- QQ-Plot 2: Student-t fordeling ---
stats.probplot(afkast, dist=stats.t, sparams=(df_t,), plot=ax_qq2)

# Sørger for at der ikke er uønsket tekst, og sætter vores egne titler
ax_qq2.set_title(f"QQ-Plot: t-fordeling (df={df_t:.2f})", fontweight='bold', fontsize=14)
ax_qq2.set_xlabel("Teoretiske Kvantiler", fontsize=12)
ax_qq2.set_ylabel("Empiriske Kvantiler", fontsize=12)

# Formatering af prikker og streg
prikker_t, streg_t = ax_qq2.get_lines()
prikker_t.set_markerfacecolor('black')
prikker_t.set_markeredgecolor('black')
prikker_t.set_alpha(0.5)
streg_t.set_color('red')
streg_t.set_linewidth(2)

ax_qq2.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout(w_pad=3.0)
plt.show()
fig2.savefig(os.path.join(grafer_mappe, "QQ_Plots_Sammenligning.png"), dpi=300, bbox_inches='tight')
