# Bachelorprojekt: Value at Risk (Python-kode)

Dette repository indeholder en oversigt over anvendte biblioteker og en kort forklaring af hvert scipts funktion.  

## 📦 Påkrævede pakker og biblioteker
Vi har samlet en liste over alle de nødvendige biblioteker i filen `requirements.txt`. 

## 📊 Bemærkning om data
Det bagvedliggende datasæt (`spx_total_return.csv`), som anvendes i projektet, er **ikke** inkluderet i dette offentlige repository (styret via `.gitignore`). For at køre koden lokalt, kræves en tilsvarende CSV-fil placeret i projektets rodmappe.

---

## 🛠️ Oversigt over scripts

Her er en funktionel oversigt over de scripts, der er anvendt i projektet:

### 1. Dataanalyse og præliminære tests
* `stationaritet.py`: Tester om tidsserien (log-afkast) er stationær ved brug af en Augmented Dickey-Fuller (ADF) test.
* `autokorrelation.py`: Tester for autokorrelation i tidsserien ved hjælp af en Ljung-Box test.
* `plot.py`: Genererer Q-Q plots af log-afkastene op imod de valgte fordelingsantagelser (Normalfordelingen og t-fordelingen) 

### 2. Modelbygning og estimation (1-dags horisont)
* `1_dags VaR.py`: Beregner de estimerede VaR- og ES-værdier for både de statiske modeller og GARCH-modellerne. Scriptet plotter desuden Q-Q plots for de standardiserede residualer og plotter VaR- og ES-estimaterne, og udfører en vægtet Ljung-Box test på de standardiserede residualer.

### 3. Evaluering og Backtesting
* `parametrisk rullende vindue.py`: Udfører Kupiec's (POF) og Christoffersens (independence) backtests på de to statiske modeller og viser resultaterne grafisk.
* `GARCH_begge.py`: Udfører Kupiec's og Christoffersens backtests for de to GARCH-modeller og viser resultaterne grafisk.
* `Historisk_VaR.py`: Beregner VaR og ES baseret på Historisk Simulation samt kører Kupiec's og Christoffersens backtests med tilhørende grafisk visualisering. 

### 4. Monte Carlo-simulering og tids-aggregering (10-dages horisont)
* `MCS.py`: Udregner 10-dages VaR og ES analytisk for Statisk-Normal-modellen. Udfører desuden Monte Carlo Simulation (MCS) med 10.000 gentagelser for at simulere 10-dages afkast for alle projektets 5 modeller, og beregner efterfølgende VaR og ES på baggrund af de simulerede fordelinger. 
