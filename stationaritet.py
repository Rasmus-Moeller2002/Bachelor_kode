import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

#Indlæs data
data =  pd.read_csv("spx_total_return.csv", index_col=0)

# 1. Definerer 3 tidsrækker
priser = data['SPXT INDEX']
simple_returns = priser.pct_change().dropna()
log_returns = np.log(priser).diff().dropna()

# 2. Funktion til at køre ADF-testen
def test_stationarity(timeseries, name):
    print(f"--- Augmented Dickey-Fuller Test for: {name} ---")
    
    # Kør testen ADF-testen (autolag='AIC' lader Python finde det optimale antal lags)
    result = adfuller(timeseries, autolag='AIC')
    
    adf_statistic = result[0]
    p_value = result[1]
    
    print(f"ADF Test Statistik: {adf_statistic:.4f}")
    print(f"P-værdi: {p_value:f}")
    
    print("Kritiske værdier:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
        
    # Konklusion baseret på 5% signifikansniveau
    if p_value <= 0.05:
        print("Konklusion: Data ER STATIONÆRT (Forkaster nulhypotesen)\n")
    else:
        print("Konklusion: Data er IKKE STATIONÆRT (Kan ikke forkaste nulhypotesen)\n")

# 3. Kør testen på alle tre tidsrækker
test_stationarity(priser, "Total Return (SPXT INDEX)")
test_stationarity(simple_returns, "Simple Afkast")
test_stationarity(log_returns, "Log-Afkast")