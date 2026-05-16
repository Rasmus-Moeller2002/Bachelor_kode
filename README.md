# Bachelorprojekt: Value at Risk (Python-kode)

Dette repository indeholder en oversigt over anvendte biblioteker og en kort forklaring af hvert scipts funktion.  

## 📦 Påkrævede pakker og biblioteker
Vi har samlet en liste over alle de nødvendige biblioteker i filen `requirements.txt`. 
For at installere alle pakker på én gang, kan du køre følgende kommando i din terminal:
`pip install -r requirements.txt`

## 📊 Bemærkning om data
Det bagvedliggende datasæt (`spx_total_return.csv`), som anvendes i projektet, indeholder fortrolige finansielle data og er derfor **ikke** inkluderet i dette offentlige repository (styret via `.gitignore`). For at køre koden lokalt, kræves en tilsvarende CSV-fil placeret i projektets rodmappe.

---

## 🛠️ Oversigt over scripts

Her er en kronologisk og funktionel oversigt over de scripts, der er anvendt i projektet:

### 1. Dataanalyse og præliminære tests
* `stationaritet.py`: Tester om tidsserien (log-afkast) er stationær ved brug af en Augmented Dickey-Fuller (ADF) test.
* `autokorrelation.py`: Tester for autokorrelation i tidsserien ved hjælp af en Ljung-Box test.
* `plot.py`: Genererer Q-Q plots af log-afkastene op imod de valgte fordelingsantagelser (Normalfordelingen og Student's t-fordelingen) for at illustrere "fat tails".

### 2. Modelbygning og estimation (1-dags horisont)
* `1_dags VaR.py`: Beregner de estimerede VaR- og ES-værdier for både de statiske modeller og GARCH-modellerne. Scriptet plotter desuden resultaterne, danner Q-Q plots for de standardiserede residualer og udfører en vægtet Ljung-Box test på residualerne.

### 3. Evaluering og Backtesting
* `parametrisk rullende vindue.py`: Eksekverer Kupiec's (POF) og Christoffersens (independens) backtests på de to statiske modeller og visualiserer resultaterne grafisk.
* `GARCH_begge.py`: Eksekverer Kupiec's og Christoffersens backtests for de to estimerede GARCH-modeller og viser resultaterne grafisk.
* `Historisk_VaR.py`: Beregner VaR og ES baseret på Historisk Simulation (HS) samt kører Kupiec's og Christoffersens backtests med tilhørende grafisk visualisering.

### 4. Avanceret simulering og tids-aggregering (10-dages horisont)
* `MCS.py`: Udregner 10-dages VaR og ES analytisk for den Statiske Normal-model. Udfører desuden Monte Carlo Simulation (MCS) med 10.000 gentagelser for at simulere 10-dages afkast for alle projektets 5 modeller, og beregner efterfølgende VaR og ES på baggrund af de simulerede fordelinger.
