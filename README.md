# chess_performance
A repo to calculate the rolling chess performance of chess.come players, by country over the history of their games.


**Chess Performance Analysis (GB Players)**

**Overview**

This project investigates the evolution of chess performance metrics—Elo rating and move accuracy—among 1,000 British Chess.com players over time. We analyze how major national events (e.g., Brexit, lockdowns, elections) coincide with changes in cognitive performance as inferred from chess data.

---

## Data Collection

* **Players**: First 1,000 UK-based public profiles via the Chess.com Public API (`/country/GB/players`).
* **Games**: Monthly archives fetched through `/player/{username}/games/archives` and `/games/{YYYY}/{MM}` endpoints.
* **Metrics per Game**:

  * **Elo Rating** (pre-game rating).
  * **Accuracy** score from engine analysis (when available).
  * **Timestamp** of game completion.

Data saved as CSV:

```
Data/random_1k_GB.csv
```

---

## Methodology

1. **Preprocessing**:

   * Parse timestamps into datetime.
   * Filter out entries without ratings.
2. **Rolling Average Elo**:

   * Compute a 10-game rolling mean per player.
3. **Monthly Aggregation**:

   * Unique active players.
   * Average games played per player.
   * Mean and standard deviation of rolling Elo.
   * Mean and standard deviation of accuracy.
4. **Event Overlay**:

   * Annotate plots with key dates (e.g., Brexit referendum, national lockdowns, elections).

---

## Figures

&#x20;*Figure 1: Number of unique active players per month.*

&#x20;*Figure 2: Average monthly games per player.*

&#x20;*Figure 3: Mean 10-game rolling Elo rating across all players.*

&#x20;*Figure 4: Standard deviation of rolling Elo ratings.*

&#x20;*Figure 5: Mean move accuracy over time.*

&#x20;*Figure 6: Standard deviation of move accuracy.*

---

## Usage

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```
3. Run data collection and analysis script:

   ```bash
   python3 analysis.py
   ```

---

## Requirements

* Python 3.8+
* `requests`, `pandas`, `matplotlib`, `tqdm`

---

