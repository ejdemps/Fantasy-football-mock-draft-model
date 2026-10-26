# Fantasy Football Draft Pick Model
This project began with the question, "What actually determines fantasy football draft picks?"
It replays five seasons of a 12-team Sleeper league's drafts (2022–2026, 966 modeled picks) and
fits discrete-choice models that predict which player each manager takes,
using only what that manager could see when they were on the clock.

The models are written in pure Python with no dependencies. 

## Approach

### 1. Data (`fetch_data.py`)
Pulls from the Sleeper API:
- every pick from the user's completed drafts (who took whom, and when)
- Sleeper's pre-draft ADP (average draft position) for each season
- each top-200 player's stats from the previous season

Players are always tagged with the NFL team they started that season on.
Other managers' names and Sleeper IDs are replaced with labels like
"Manager A" for their privacy (pass `--keep-names` to skip this).

### 2. Conditional logit pick model (`pick_model.py`)
At every pick, the manager chooses one player from a **choice set**: the 25
best remaining players by ADP, plus whoever was actually taken. Each candidate
is described by 20 traits in four groups:

| Group | Examples |
| --- | --- |
| Player | ADP value (has the player fallen?), rank among players left, prior-season PPG and games played |
| Roster / draft position | open starting slot at the position, players already rostered there, whether the player will last until the manager's next pick, position runs |
| League | how often the league spends this round's picks on the position |
| Manager habits | this manager's lean toward a position at this stage of the draft, and toward players from a favorite NFL team |

Manager habits are shrunk toward the league average with pseudo-counts and
measured on all other seasons (i.e. 2024 habits are explained without 
including 2024 choices)

Trait importance is reported as standardized coefficients, z-scores, and
permutation importance.

### 3. Two-stage model (`two_stage_model.py`)
Factors each pick as

```
P(player) = P(position) × P(player | position)
```

Step 1 picks a position, starting from the league's historical share for that
round (as a fixed offset) and adjusting for roster needs, runs and scarcity.
Step 2 ranks the best remaining players at that position. Splitting the
decision keeps predicted position totals in line with how the league actually
drafts.

### Evaluation
Every accuracy number is out of sample: each season is predicted by a
model fit only on the other seasons.

## Results

Exact pick prediction, all rounds:

| Model | Top-1 | Top-3 |
| --- | --- | --- |
| Baseline: best available by ADP | 17.4% | — |
| Conditional logit (+ league position prefs) | 22.4% | 48.4% |
| Two-stage (+ manager habits) | 24.4% | 47.7% |

Predicting the **position** of the pick, all rounds:

| Model | Top-1 | Top-2 |
| --- | --- | --- |
| Baseline: best available's position | 30.0% | — |
| Baseline: league's most common position for the round | 36.7% | — |
| Two-stage | 48.0% | 74.6% |

In rounds 1–2 the two-stage model names the exact player about 56% of the
time. The two-stage model also tracks the league's cumulative position counts
much better than the single-stage logit: it misses by 0.48 players per position
in rounds 6–10, versus 1.69.

Strongest drivers of a pick (all rounds, by permutation importance): rank among
remaining players, ADP value, position (QB and DEF are taken late), and an open
starting slot. Manager-specific habits are statistically significant but small.

Full tables are in [`results/`](results/).

## Project structure

```
fetch_data.py          Download picks, ADP and prior-season stats from Sleeper
pick_model.py          Conditional logit model, trait importance, manager tendencies
two_stage_model.py     Position-then-player model
data/                  Input CSVs built by fetch_data.py
results/               Model outputs
```

| Output | Contents |
| --- | --- |
| `selection_trait_importance.csv` | coefficient, z, odds ratio and permutation importance of each trait |
| `early_round_trait_importance.csv` | the same, fit on rounds 1–5 only |
| `round_accuracy.csv` / `position_accuracy.csv` | out-of-sample player and position accuracy by draft stage |
| `position_pace.csv` / `two_stage_pace.csv` | cumulative players per position by round, actual vs. predicted |
| `two_stage_accuracy.csv` | two-stage model accuracy by draft stage |
| `manager_position_tendencies.csv` | each manager's position mix by draft stage vs. the league |
| `manager_team_affinity.csv` | over-drafted NFL teams per manager, with Poisson p-values |

## Running it

Requires Python 3.8+ and nothing else.

```bash
python fetch_data.py <sleeper_username>   # optional: data/ is already included
python pick_model.py                      # ~1 minute
python two_stage_model.py                 # ~1 minute
```
