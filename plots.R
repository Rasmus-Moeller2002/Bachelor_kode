rm(list = ls())

# ==============================================================================
# 1. Installer og indlæs kun de nødvendige pakker
# (Slet # foran install.packages, hvis du skal installere dem første gang)
# ==============================================================================
# install.packages("ggplot2")
# install.packages("patchwork")
# install.packages("forecast")

library(ggplot2)
library(patchwork)
library(forecast)

# ==============================================================================
# 2. Hent selve vektor-værdierne 
# (Forudsætter at 'spx_total_return' allerede er indlæst i dit environment)
# ==============================================================================
priser <- spx_total_return$SPXT.INDEX

# ==============================================================================
# 3. Sæt tids-parametre op
# ==============================================================================
startyear <- 2018  # Startår
startday <- 1      # Første observation i året 
T <- length(priser) 
s <- 252           # 252 handelsdage på et børs-år
years <- T/s       # Hvor mange års data i alt

# ==============================================================================
# 4. Lav data om til et timeseries (ts) objekt
# ==============================================================================
spx_ts <- ts(priser, start = c(startyear, startday), frequency = s)
afkast <- na.omit(diff(log(spx_ts))) *100
mu <- mean(afkast)
std <- sqrt(var(afkast))
at <- afkast - mu

# ==============================================================================
# 5. Stationær VS ikke stationær proces plots
# ==============================================================================
# Byg data frame for priser (og tilføj indeksering)
df_priser <- data.frame(
  Tid = as.numeric(time(spx_ts)), 
  Pris = as.numeric(spx_ts)
)

# Indekser priserne, så de starter i 1
første_pris <- df_priser$Pris[1] 
df_priser$Indekseret_Pris <- df_priser$Pris / første_pris

df_afkast <- data.frame(
  Tid = as.numeric(time(afkast)), 
  Afkast = as.numeric(afkast)
)

# Byg det første plot: Indekseret Total Return
figur1 <- ggplot(data = df_priser, aes(x = Tid, y = Indekseret_Pris)) +
  geom_line(color = "darkblue", linewidth = 0.7) +
  labs(
    title = "S&P 500: Total Return Indeks", 
    x = "År",
    y = "Indeksværdi (Start = 1)"          
  ) +
  theme_minimal() + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# Byg det andet plot: Log-afkast 
figur2 <- ggplot(data = df_afkast, aes(x = Tid, y = Afkast)) +
  geom_line(color = "darkred", linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") + # Viser 0-linjen
  labs(
    title = "S&P 500: Log-afkast",
    x = "År",
    y = "Dagligt log-afkast"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# Sæt dem flot ved siden af hinanden med 'patchwork' pakken
samlet_figur <- figur1 + figur2

# Vis det endelige plot
print(samlet_figur)

# ==============================================================================
# 6. ACF plots
# ==============================================================================
# Vi bruger 'ggAcf' fra forecast-pakken og 'as.numeric' for at få heltal på x-aksen
acf1 <- ggAcf(as.numeric(spx_ts), lag.max = 50) +
  theme_minimal() +
  labs(title = "ACF: Total Return", y = "ACF") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

acf2 <- ggAcf(as.numeric(afkast), lag.max = 20) +
  theme_minimal() +
  labs(title = "ACF: Log-afkast", y = "ACF") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

samlet_2x2_figur <- acf1 + acf2
print(samlet_2x2_figur)
