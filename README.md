# chess_performance
A repo to calculate the rolling chess performance of chess.come players, by country over the history of their games.


**Chess Performance Analysis (GB Players)**

**Overview**

This project ultimately aims to analyse the full rating history of all chess players - to determine the effect of national-level events on cognitive performance. For now, we have pulled the full game histories of 1k players registered with the nationality of GB (out of potential 6m). Just to test how it works with (e.g., Brexit, lockdowns, elections).

**Progress and next steps**
It took 4 hours to pull 1k users full chess histories using this code. It can be sped up for sure, running in parallel etc. But probably we need a virtual machine to take on this effort.

Next step: a) run it on a virtual machine; b) upgrade to entire GB userbase; c) plot and reflect how to expand

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

