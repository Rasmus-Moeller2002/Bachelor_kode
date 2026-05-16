import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf # TILFØJET TIL DEN VÆGTEDE TEST

from arch import arch_model
import os

# --- NY FUNKTION TIL VÆGTET LJUNG-BOX TEST ---
def weighted_ljung_box_sq(x, lags):
    """
    Weighted Ljung-Box test for squared residuals using analytical 
    Gamma approximations (alpha and beta) from Fisher & Gallagher (2012).
    """
    n = len(x)
    max_lag = max(lags)
    r = acf(x, nlags=max_lag, fft=True)[1:] 
    results = []
    
    for m in lags:
        # 1. Udregn test-statistikken Q_W* (Ligning 12 i papiret)
        k = np.arange(1, m + 1)
        weights = (m - k + 1) / m
        q_terms = (weights * (r[:m]**2)) / (n - k)
        q_stat = n * (n + 2) * np.sum(q_terms)
        
        # 2. Analytisk formel for Gamma-fordelingen Shape (alpha)
        alpha = (3 * m * (m + 1)) / (8 * m + 4)
        
        # Analytisk formel for Gamma-fordelingen scale (beta)
        beta_scale = (2 * (2 * m + 1)) / (3 * m)
        
        # 3. Udregn P-værdi (Højre hale af Gamma-fordelingen)
        # Vi bruger stats.gamma.sf (Survival Function), som er det samme som 1 - cdf
        p_val = stats.gamma.sf(q_stat, a=alpha, scale=beta_scale)
            
        results.append({
            'Lags': m, 
            'Test-Statistik (Q_W*)': q_stat, 
            'P-værdi': p_val,
            'Alpha (Shape)': alpha,
            'Beta (Scale)': beta_scale
        })
        
    df_results = pd.DataFrame(results).set_index('Lags')
    
    # Formatering af P-værdier
    df_results['P-værdi'] = df_results['P-værdi'].apply(
        lambda val: '< 2.2e-16' if val < 2.2e-16 else f"{val:.6f}"
    )
    
    return df_results

def weighted_ljung_box_mean(x, lags, dof=0):
    """
    Vægtet Ljung-Box test til ALMINDELIGE standardiserede residualer.
    Bruger Ligning 5, 7 og 8 fra Fisher & Gallagher (2012).
    'dof' er (p+q) fra ARMA-modellen. Default er 0 for ARMA(0,0).
    """
    n = len(x)
    max_lag = max(lags)
    # Autokorrelation af de almindelige residualer
    r = acf(x, nlags=max_lag, fft=True)[1:] 
    results = []
    
    for m in lags:
        # 1. Udregn test-statistikken Q_W (Ligning 5)
        k = np.arange(1, m + 1)
        weights = (m - k + 1) / m
        q_terms = (weights * (r[:m]**2)) / (n - k)
        q_stat = n * (n + 2) * np.sum(q_terms)
        
        # 2. Analytiske formler for Gamma-fordelingen (Ligning 7 og 8 fra papiret)
        # Shape (alpha) - Ligning 7
        numerator_alpha = 3 * m * (m + 1)**2
        denominator_alpha = 4 * (2 * m**2 + 3 * m + 1 - 6 * m * dof)
        alpha = numerator_alpha / denominator_alpha
        
        # Scale (beta) - Ligning 8
        numerator_beta = 2 * (2 * m**2 + 3 * m + 1 - 6 * m * dof)
        denominator_beta = 3 * m * (m + 1)
        beta_scale = numerator_beta / denominator_beta
        
        # 3. Udregn P-værdi (Højre hale af Gamma-fordelingen)
        p_val = stats.gamma.sf(q_stat, a=alpha, scale=beta_scale)
            
        results.append({
            'Lags': m, 
            'Test-Statistik (Q_W)': q_stat, 
            'P-værdi': p_val,
            'Alpha (Shape)': alpha,
            'Beta (Scale)': beta_scale
        })
        
    df_results = pd.DataFrame(results).set_index('Lags')
    
    # Formatering af P-værdier
    df_results['P-værdi'] = df_results['P-værdi'].apply(
        lambda val: '< 2.2e-16' if val < 2.2e-16 else f"{val:.6f}"
    )
    
    return df_results

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

konfidensniveau = 0.95
c_alpha = 1 - konfidensniveau

print("\n" + "="*60)
print(f" ENDELIG 1-DAGS VaR BEREGNING FOR I MORGEN ({konfidensniveau*100}%)")
print("="*60)

# ==========================================
# 1. PARAMETRISK VaR og ES
# ==========================================
mu_statisk = np.mean(afkast)
sigma_statisk = np.std(afkast)

# Normalfordelt Statisk VaR
z_score = stats.norm.ppf(c_alpha)
var_param_norm = (mu_statisk + z_score * sigma_statisk)

#Normalfordelt Statisk ES
pdf_norm = stats.norm.pdf(z_score)
ES_norm = (mu_statisk - sigma_statisk * (pdf_norm / c_alpha))

# Student-t Statisk VaR
df_statisk = stats.t.fit(afkast)[0]
t_score = stats.t.ppf(c_alpha, df=df_statisk)

# Vi omregner standardafvigelse (sigma) til skaleringsparameteren (lambda)
# Formel fra side 95 i statistikbogen
lambda_scale = sigma_statisk * np.sqrt((df_statisk - 2) / df_statisk)
var_param_t = (mu_statisk + lambda_scale * t_score)

#Student-t Statisk ES
pdf_t = stats.t.pdf(t_score, df=df_statisk)

#Udregn brøken for de tykke haler
hale_straf = (t_score**2 + df_statisk) / (df_statisk - 1)
ES_t = (mu_statisk - lambda_scale * (pdf_t / c_alpha) * hale_straf)

print("\n--- STATISK PARAMETRISK VaR og ES ---")
print(f"Gennemsnit (mu): {mu_statisk:.4f}%")
print(f"Standardafvigelse (sigma): {sigma_statisk:.4f}%")
print(f"Estimeret df for t-fordeling: {df_statisk:.2f}")
print(f"-> Normal VaR: {var_param_norm:.4f}%")
print(f"-> Student-t VaR: {var_param_t:.4f}%")
print(f"-> Normal ES: {ES_norm:.4f}%")
print(f"-> Student-t ES: {ES_t:.4f}%")

# --- TILFØJELSE: Pre-diagnostik for ARCH-effekter (Før GARCH) ---
print("\n" + "="*60)
print(" PRE-DIAGNOSTIK: Test for volatilitetsklynger i markedsstød")
print("="*60)

# Udregner markedsstødet (a_t) 
a_t = afkast - mu_statisk

# Kvadrerer stødene for at gøre klar til at teste for afhængighed i variansen
a_t_sq = a_t ** 2

# Udfører den NORMALE Ljung-Box test (Vi bruger statsmodels her, da sigma ikke er estimeret endnu)
print("\n--- Ljung-Box Test på rå kvadrerede stød ---")
lb_test_pre = acorr_ljungbox(a_t_sq, lags=[5, 10, 20], model_df=0, return_df=True)

# Formaterer outputtet, så det matcher resten af jeres script
lb_test_pre.columns = ['Test-Statistik (Q)', 'P-værdi']
lb_test_pre.index.name = 'Lags'
lb_test_pre['P-værdi'] = lb_test_pre['P-værdi'].apply(lambda x: '< 2.2e-16' if x < 2.2e-16 else f"{x:.6f}")
print(lb_test_pre)

# Hjælpetekst til fortolkningen
print("\nFortolkning (Pre-diagnostik):")
print("H0: Der er INGEN volatilitetsklynger (Data er hvid støj).")
print("H1: Der ER volatilitetsklynger i dataene.")
print("-> Vi forventer/ønsker en P-værdi UNDER 0.05, så vi KAN afvise H0 og dermed retfærdiggøre brugen af GARCH!")

# ==========================================
# 2. GARCH(1,1) VaR (Dynamisk - betinget af i dag)
# ==========================================
print("\n" + "="*60)
print(" DYNAMISK GARCH(1,1) VaR (Trænet på HELE datasættet)")
print("="*60)

# ------------------------------------------
# A) GARCH Normal
# ------------------------------------------
am_norm = arch_model(afkast, mean='Constant', vol='GARCH', p=1, q=1, dist='Normal')
res_norm = am_norm.fit(disp='off')
forecast_norm = res_norm.forecast(horizon=1)

mu_garch_norm = forecast_norm.mean.iloc[-1].values[0]
sigma_garch_norm = np.sqrt(forecast_norm.variance.iloc[-1].values[0])
var_garch_norm = (mu_garch_norm + z_score * sigma_garch_norm)

#ES for normalfordelt GARCH
es_garch_norm = (mu_garch_norm - sigma_garch_norm * (pdf_norm / c_alpha))

print("\n--- GARCH(1,1) NORMAL PARAMETRE ---")
print(res_norm.summary())

std_resid_norm = res_norm.std_resid.dropna()
std_resid_norm_sq = std_resid_norm ** 2

# ÆNDRET TIL VÆGTET TEST
print("\n--- Vægtet Ljung-Box Test NORMAL (Er volatilitetsklyngerne fjernet af GARCH?) ---")
lb_test_post_norm = weighted_ljung_box_sq(std_resid_norm_sq, lags=[5, 10, 20])
print(lb_test_post_norm)

# ÆNDRET TIL VÆGTET TEST
print("\n--- Vægtet Ljung-Box Test NORMAL (Er der autokorrelation i middelværdien?) ---")
lb_test_mean_norm = weighted_ljung_box_mean(std_resid_norm, lags=[5, 10, 20])
print(lb_test_mean_norm)

# En lille hjælpende tekst til fortolkningen:
print("\nFortolkning (Post-diagnostik for Normal):")
print("H0: Der er INGEN autokorrelation tilbage (GARCH-Normal er en succes!).")
print("H1: Der ER autokorrelation tilbage (Modellen fanger ikke alt).")
print("-> Vi ønsker en P-værdi OVER 0.05, så vi IKKE kan afvise H0.")

# ------------------------------------------
# B) GARCH Student-t
# ------------------------------------------
am_t = arch_model(afkast, mean='Constant', vol='Garch', p=1, q=1, dist='StudentsT')
res_t = am_t.fit(disp='off')
forecast_t = res_t.forecast(horizon=1)

mu_garch_t = forecast_t.mean.iloc[-1].values[0]
sigma_garch_t = np.sqrt(forecast_t.variance.iloc[-1].values[0])
df_garch = res_t.params['nu']

# --- Skalering for GARCH t-fordeling ---
t_score_garch = stats.t.ppf(c_alpha, df=df_garch)

# Vi omregner den forudsagte GARCH standardafvigelse (sigma) til skaleringsparameteren (lambda)
#Formel fra side 95 i statistikbogen
lambda_garch = sigma_garch_t * np.sqrt((df_garch - 2) / df_garch)
var_garch_t = (mu_garch_t + lambda_garch * t_score_garch)

#ES for Student-t fordeling GARCH
# Henter tætheden ved t-scoren
pdf_t_garch = stats.t.pdf(t_score_garch, df=df_garch)
# Regner "straffen" for de tykke haler ud
hale_straf_garch = (t_score_garch**2 + df_garch) / (df_garch - 1)
# Regner 1-dags GARCH ES ud
es_garch_t = (mu_garch_t - lambda_garch * (pdf_t_garch / c_alpha) * hale_straf_garch)


print("\n--- GARCH(1,1) STUDENT-t PARAMETRE ---")
print(res_t.summary())

std_resid_t = res_t.std_resid.dropna()
std_resid_t_sq = std_resid_t ** 2

# ÆNDRET TIL VÆGTET TEST
print("\n--- Vægtet Ljung-Box Test t-fordeling (Er volatilitetsklyngerne fjernet af GARCH?) ---")
lb_test_post_t = weighted_ljung_box_sq(std_resid_t_sq, lags=[5, 10, 20])
print(lb_test_post_t)

# ÆNDRET TIL VÆGTET TEST
print("\n--- Vægtet Ljung-Box Test t-fordeling (Er der autokorrelation i middelværdien?) ---")
lb_test_mean_t = weighted_ljung_box_mean(std_resid_t, lags=[5, 10, 20])
print(lb_test_mean_t)

# En lille hjælpende tekst til fortolkningen af standardiserede kvadrerede residualer:
print("\nFortolkning (Post-diagnostik):")
print("H0: Der er INGEN autokorrelation tilbage (GARCH-modellen er en succes!).")
print("H1: Der ER autokorrelation tilbage (Modellen fanger ikke alt).")
print("-> Vi ønsker en P-værdi OVER 0.05, så vi IKKE kan afvise H0.")


# ------------------------------------------
# C) Print Endelige VaR Resultater
# ------------------------------------------
print(f"\n--- ENDELIGE GARCH FORECASTS FOR I MORGEN ({konfidensniveau*100}%) ---")
print(f"Forudsagt volatilitet (Normal): {sigma_garch_norm:.4f}%")
print(f"Forudsagt volatilitet (Student-t): {sigma_garch_t:.4f}% (df={df_garch:.2f})")
print(f"-> GARCH Normal VaR: {var_garch_norm:.4f}%")
print(f"-> GARCH Student-t VaR: {var_garch_t:.4f}%")
print(f"-> NORMAL ES for GARCH: {es_garch_norm:.4f}%")
print(f"-> Student-t ES for GARCH: {es_garch_t:.4f}%")

# ==========================================
# 3. SAMLET VISUALISERING (Opdelt i separate plots)
# ==========================================

# Definer x-aksen til de teoretiske kurver
x_axis = np.linspace(afkast.min(), afkast.max(), 1000)

# Udregn tætheden for hele kurven (til Plot 1)
pdf_n_curve = stats.norm.pdf(x_axis, loc=mu_statisk, scale=sigma_statisk)
pdf_t_curve = stats.t.pdf(x_axis, df=df_statisk, loc=mu_statisk, scale=lambda_scale)

# ---------------------------------------------------------
# PLOT 1: Statisk Parametrisk VaR + Teoretiske Fordelinger
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(afkast, bins=150, density=True, alpha=0.4, color='steelblue', label='Empirisk Fordeling (S&P 500)')

# 1. Tegn de teoretiske kurver
plt.plot(x_axis, pdf_n_curve, color='olivedrab', linestyle='--', linewidth=2, alpha=0.8, label='Normalfordeling')
plt.plot(x_axis, pdf_t_curve, color='purple', linestyle='--', linewidth=2, alpha=0.8, label=f't-fordeling (df={df_statisk:.2f}))')

plt.title(f"Tilpasning af normal- og t-fordeling til daglige log-afkast", fontsize=14, fontweight='bold')
plt.xlabel("Dagligt log-afkast (%)")
plt.ylabel("Tæthed")
plt.legend(loc='upper left', framealpha=0.9)
plt.xlim(-6, 4) 
plt.grid(axis='y', alpha=0.3)

# Gem Plot 1
gem_sti_param = os.path.join(grafer_mappe, "Statisk_VaR_og_Fordelinger.png")
plt.savefig(gem_sti_param, dpi=300, bbox_inches='tight')
#plt.yscale('log')
plt.show()


# ---------------------------------------------------------
# PLOT 2: Dynamisk GARCH VaR Forecast
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(afkast, bins=150, density=True, alpha=0.4, color='steelblue', label='Empirisk Fordeling (S&P 500)')

# Tegn VaR grænserne for GARCH
plt.axvline(x=var_garch_norm, color='blue', linestyle='-', linewidth=2.5, label=f'GARCH-Normal VaR: {var_garch_norm:.2f}%')
plt.axvline(x=var_garch_t, color='darkorange', linestyle='-', linewidth=2.5, label=f'GARCH-t VaR: {var_garch_t:.2f}%')

plt.title(f"Dynamisk GARCH(1,1) 1-dags VaR ({konfidensniveau*100}%)", fontsize=14, fontweight='bold')
plt.xlabel("Dagligt log-afkast (%)")
plt.ylabel("Tæthed")
plt.legend(loc='upper left', framealpha=0.9)
plt.xlim(-6, 4) 
plt.grid(axis='y', alpha=0.3)

# Gem Plot 2
gem_sti_garch = os.path.join(grafer_mappe, "GARCH_VaR_Forecast.png")
plt.savefig(gem_sti_garch, dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------
# PLOT 3: Statisk Parametrisk VaR og Expected Shortfall (ES)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(afkast, bins=150, density=True, alpha=0.4, color='steelblue', label='Empirisk Fordeling (S&P 500)')

# Normalfordeling: VaR og ES
plt.axvline(x=var_param_norm, color='olivedrab', linestyle='--', linewidth=2, label=f'Statisk-Normal VaR: {var_param_norm:.2f}%')
plt.axvline(x=ES_norm, color='olivedrab', linestyle=':', linewidth=3.5, label=f'Statisk-Normal ES: {ES_norm:.2f}%')

# Student-t fordeling: VaR og ES
plt.axvline(x=var_param_t, color='purple', linestyle='--', linewidth=2, label=f'Statisk-t VaR: {var_param_t:.2f}%')
plt.axvline(x=ES_t, color='purple', linestyle=':', linewidth=3.5, label=f'Statisk-t ES: {ES_t:.2f}%')

plt.title(f"Statisk 1-dags Risiko ({konfidensniveau*100}%): VaR vs. ES", fontsize=14, fontweight='bold')
plt.xlabel("Dagligt log-afkast (%)")
plt.ylabel("Tæthed")
plt.legend(loc='upper left', framealpha=0.9)
plt.xlim(-6, 4) 
plt.grid(axis='y', alpha=0.3)

# Gem Plot 3
gem_sti_param_es = os.path.join(grafer_mappe, "Statisk_VaR_og_ES.png")
plt.savefig(gem_sti_param_es, dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------
# PLOT 4: Dynamisk GARCH VaR og Expected Shortfall (ES)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(afkast, bins=150, density=True, alpha=0.4, color='steelblue', label='Empirisk Fordeling (S&P 500)')

# GARCH Normal: VaR og ES
plt.axvline(x=var_garch_norm, color='blue', linestyle='-', linewidth=2, label=f'GARCH-Normal VaR: {var_garch_norm:.2f}%')
plt.axvline(x=es_garch_norm, color='blue', linestyle=':', linewidth=3.5, label=f'GARCH-Normal ES: {es_garch_norm:.2f}%')

# GARCH Student-t: VaR og ES
plt.axvline(x=var_garch_t, color='darkorange', linestyle='-', linewidth=2, label=f'GARCH-t VaR: {var_garch_t:.2f}%')
plt.axvline(x=es_garch_t, color='darkorange', linestyle=':', linewidth=3.5, label=f'GARCH-t ES: {es_garch_t:.2f}%')

plt.title(f"Dynamisk GARCH(1,1) 1-dags Risiko ({konfidensniveau*100}%): VaR vs. ES", fontsize=14, fontweight='bold')
plt.xlabel("Dagligt log-afkast (%)")
plt.ylabel("Tæthed")
plt.legend(loc='upper left', framealpha=0.9)
plt.xlim(-6, 4) 
plt.grid(axis='y', alpha=0.3)

# Gem Plot 4
gem_sti_garch_es = os.path.join(grafer_mappe, "GARCH_VaR_og_ES.png")
plt.savefig(gem_sti_garch_es, dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# --- NYT: PLOT 5: Q-Q Plots af Standardiserede Residualer ---
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Q-Q for GARCH(1,1) Normalfordeling
sm.qqplot(std_resid_norm, dist=stats.norm, fit=True, line='45', ax=axes[0], 
          markerfacecolor='black', markeredgecolor='black', alpha=0.5)
axes[0].set_title("GARCH-Normal standardiserede residualer", fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
# --- NYT: Danske akser til Plot 1 ---
axes[0].set_xlabel("Teoretiske Kvantiler", fontsize=11)
axes[0].set_ylabel("Empiriske Kvantiler", fontsize=11)

# Plot 2: Q-Q for GARCH(1,1) Student-t fordeling
# Matematisk vigtig skalering af den teoretiske t-fordeling i plottet:
scale_factor_qq = np.sqrt((df_garch - 2) / df_garch)
sm.qqplot(std_resid_t, dist=stats.t, distargs=(df_garch,), loc=0, scale=scale_factor_qq, 
          line='45', ax=axes[1], markerfacecolor='black', markeredgecolor='black', alpha=0.5)
axes[1].set_title(f"GARCH-t standardiserede residualer (df={df_garch:.2f})", fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
# --- NYT: Danske akser til Plot 2 ---
axes[1].set_xlabel("Teoretiske Kvantiler", fontsize=11)
axes[1].set_ylabel("Empiriske Kvantiler", fontsize=11)

# Gør layoutet pænt og gem
plt.tight_layout()
gem_sti_qq = os.path.join(grafer_mappe, "QQ_Plots_GARCH_Residualer.png")
plt.savefig(gem_sti_qq, dpi=300, bbox_inches='tight')
plt.show()