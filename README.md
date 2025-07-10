# chess_performance
A repo to calculate the rolling chess performance of chess.come players, by country over the history of their games.


**"Seasonal Attention Disorder": what 300m chess games reveals about global concentration patterns**

**Overview**

This project ultimately aims to analyse the chess performance of nations - to reveal something about the world through chess.

**Progress and next steps**

Edit: 10/07/25 We now have AWS scraping around 1 country per week - only selecting consistent users since 2017 and recording win %. This gives us around 1k games from 2010 onwards; and around 25-50m games per country

Next step: a) let it run for 7 countries b) analyse the data c) decide if there are other questions to answer (ie. time during day; country vs other country).

Previous step: We achieved code that pulled data for a random sample of around 5k players for 20 countries - it made pretty plots, but needed improving for analysis.

**Repo Structure**

 - Code: code to pull data and analyse data in the code folder
 - Data: Three rounds of data scraping (random 1k players UK; 5k players for 20 countries; 25m games... ongoing) 
 - Figures: silly outputs in the figures folder
 - Old structure: an old structure for code pulling involved many different python files rather than one single one we use now

Below: Silly stuff I did to start with for funzies...

---

## Data Collection

* **Players**: First 1,000 UK-based public profiles via the Chess.com Public API (`/country/GB/players`).
* **Games**: Monthly archives fetched through `/player/{username}/games/archives` and `/games/{YYYY}/{MM}` endpoints.
* **Metrics per Game**:

  * **Elo Rating** (pre-game rating).
  * **Accuracy** score from engine analysis (when available).

Data saved as CSV:

```
Data/random_1k_GB.csv
```

---

## Figures
We see, of our 1k players, many quit the app after the passing of Queen Elizabeth - as loyal subjects, out of respect and mourning for our beloved majesty.

<p align="center">
  <img src="https://raw.githubusercontent.com/BenGoodair/chess_performance/main/Figures/active_players.png"  />
</p>

We see that people start playing more games per day after the Brexit referendum in a deep effort to reconnect to our European sisters, brothers and siblings through the form of chess.

<p align="center">
  <img src="https://raw.githubusercontent.com/BenGoodair/chess_performance/main/Figures/games_per_player.png"  />
</p>

We see that people's accuracy does decline during COVID-19. Probably because people were often drunk whilst playing.

<p align="center">
  <img src="https://raw.githubusercontent.com/BenGoodair/chess_performance/main/Figures/mean_accuracy.png"  />
</p>


We see that people's rating performance declines consistently over time - a meaningful reflection of this great nations' race to the bottom of all measures of wealth, well-being and value.

<p align="center">
  <img src="https://raw.githubusercontent.com/BenGoodair/chess_performance/main/Figures/mean_rolling_elo.png"  />
</p>

Britain is a country that deeply seeks rising inequality. It is a country that believes in inequality as a fundamental national value. The austerity era provided a great rise in inequality in chess rating - as it did material wealth.
<p align="center">
  <img src="https://raw.githubusercontent.com/BenGoodair/chess_performance/main/Figures/std_rolling_elo.png"  />
</p>

---

