# chess_performance
A repo to calculate the rolling chess performance of chess.come players, by country over the history of their games.


**Chess Performance Analysis (GB Players)**

**Overview**

This project ultimately aims to analyse the full rating history of all chess players - to determine the effect of national-level events on cognitive performance. For now, we have pulled the full game histories of 1k players registered with the nationality of GB (out of potential 6m). Just to test how it works with (e.g., Brexit, lockdowns, elections).

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

&#x20;*Figure 1: Number of unique active players per month.*

&#x20;*Figure 2: Average monthly games per player.*

&#x20;*Figure 3: Mean 10-game rolling Elo rating across all players.*

&#x20;*Figure 4: Standard deviation of rolling Elo ratings.*

&#x20;*Figure 5: Mean move accuracy over time.*

&#x20;*Figure 6: Standard deviation of move accuracy.*

---



---

## Requirements

* Python 3.8+
* `requests`, `pandas`, `matplotlib`, `tqdm`

---

