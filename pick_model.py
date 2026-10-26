"""Which player traits, and which manager habits, drive draft picks?

Replays every draft in data/draft_picks.csv. At each pick, the picking manager
chose one player from a set of available players (the best CHOICE_SET_SIZE left
by ADP, plus whoever was actually taken). A conditional logit model estimates
how much each trait of a candidate raises or lowers the odds they are the pick.

Two traits are manager-specific habits learned from that manager's history:
  - owner_pos_round_lift: how much more often this manager takes this
    position in this stage of the draft than the league does
    (e.g. a manager who always opens RB-RB)
  - owner_nfl_team_lift: how much more often this manager takes players from
    this NFL team than the league does (favorite-team bias)
Habits are measured leave-one-season-out: a pick in 2024 is explained only by
the manager's other drafts, so the model is graded on predicting, not recalling.

Outputs (in results/):
  - selection_trait_importance.csv: coefficient and importance of each trait
  - early_round_trait_importance.csv: the same, fit on rounds 1-5 only
  - round_accuracy.csv: out-of-sample pick accuracy by stage of the draft
  - position_accuracy.csv: out-of-sample position accuracy by stage of the draft
  - position_pace.csv: cumulative players per position by round, actual vs. expected
  - manager_position_tendencies.csv: each manager's position mix by draft stage
  - manager_team_affinity.csv: each manager's over-drafted NFL teams

Run fetch_data.py first to build the input CSVs.
"""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

PICKS_CSV = DATA_DIR / "draft_picks.csv"
ADP_CSV = DATA_DIR / "adp_rankings.csv"
STATS_CSV = DATA_DIR / "prior_season_stats.csv"
IMPORTANCE_CSV = RESULTS_DIR / "selection_trait_importance.csv"
POSITION_TENDENCIES_CSV = RESULTS_DIR / "manager_position_tendencies.csv"
TEAM_AFFINITY_CSV = RESULTS_DIR / "manager_team_affinity.csv"
ROUND_ACCURACY_CSV = RESULTS_DIR / "round_accuracy.csv"
EARLY_IMPORTANCE_CSV = RESULTS_DIR / "early_round_trait_importance.csv"
POSITION_ACCURACY_CSV = RESULTS_DIR / "position_accuracy.csv"
POSITION_PACE_CSV = RESULTS_DIR / "position_pace.csv"
FOCUS_ROUNDS = 5

CHOICE_SET_SIZE = 25
#many leagues (this included) also uses two flexes, these were 
# exlcuded as they don't fit true "positional need" narratives
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
ROUND_BUCKETS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 99, "10+")]
NFL_TEAMS = 32
# Pseudo-counts that shrink a manager's habits toward the league average, so a
# couple of picks don't look like a strong preference.
POSITION_PRIOR_PICKS = 10
TEAM_PRIOR_PICKS = 40
RIDGE = 1.0  # L2 penalty on the logit coefficients
NEWTON_ITERS = 25
MAX_STEP_HALVINGS = 30
PERMUTATION_REPEATS = 5
SEED = 0
MIN_PROBABILITY = 1e-6  # floor for scoring picks the model all but ruled out

PLAYER_FEATURES = [
    "adp_value",          # pick number minus ADP (+ = player has fallen)
    "log_available_rank", # log of rank among players left (0 = best available)
    "prior_ppg",          # previous season PPR points per game
    "prior_gp",           # previous season games played
    "no_prior_stats",     # rookie, missed the year, or outside top-200 ADP
    "pos_count",          # players already rostered at the player's position
    "starter_need",       # team still has an open starting slot at that position
    "nfl_team_stack",     # team already has a player from the same NFL team
    "pos_QB", "pos_WR", "pos_TE", "pos_K", "pos_DEF",  # vs. RB
]
DRAFT_POSITION_FEATURES = [
    "gone_before_next_pick",  # ADP says the player won't last to this manager's next pick
    "position_run",           # picks at the position since this manager last picked
    "pos_options_next_pick",  # other candidates at the position likely left next pick
]
LEAGUE_FEATURES = [
    "league_pos_round_share",  # log share of this round's picks the league spends on the position
    "log_pos_candidates",      # log number of candidates at the position for this pick
]
OWNER_FEATURES = ["owner_pos_round_lift", "owner_nfl_team_lift"]
FEATURES = PLAYER_FEATURES + DRAFT_POSITION_FEATURES + LEAGUE_FEATURES + OWNER_FEATURES
BASE_FEATURES = PLAYER_FEATURES + DRAFT_POSITION_FEATURES
PREF_FEATURES = BASE_FEATURES + LEAGUE_FEATURES


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def load_inputs():
    picks = read_csv(PICKS_CSV)

    adp = defaultdict(list)  # season -> players sorted by ADP
    for r in read_csv(ADP_CSV):
        r["adp_ppr"] = float(r["adp_ppr"])
        adp[r["season"]].append(r)
    for players in adp.values():
        players.sort(key=lambda r: r["adp_ppr"])

    stats = {}  # (adp_season, player_id) -> previous-season stats row
    for r in read_csv(STATS_CSV):
        if r["has_stats"] == "True":
            stats[(r["adp_season"], r["player_id"])] = r
    return picks, adp, stats


def round_bucket(rnd):
    for lo, hi, label in ROUND_BUCKETS:
        if lo <= int(rnd) <= hi:
            return label


class Tendencies:
    """Manager habits from pick history, optionally leaving one season out.

    Every count is stored twice: under its key (all seasons) and under
    (season, *key), so the leave-one-season-out count is a subtraction.
    """

    def __init__(self, picks):
        self.counts = Counter()
        for p in picks:
            owner, bucket, pos, team = (p["owner"], round_bucket(p["round"]),
                                        p["position"], p["nfl_team"])
            for key in [("owner_bucket_pos", owner, bucket, pos),
                        ("owner_bucket", owner, bucket),
                        ("bucket_pos", bucket, pos),
                        ("bucket", bucket),
                        ("round_pos", p["round"], pos),
                        ("round", p["round"]),
                        ("owner_team", owner, team),
                        ("owner",  owner),
                        ("team", team),
                        ("all",)]:
                self.counts[key] += 1
                self.counts[(p["season"], *key)] += 1

    def count(self, key, exclude_season=None):
        n = self.counts[key]
        return n - self.counts[(exclude_season, *key)] if exclude_season else n

    def league_position_share(self, bucket, pos, exclude_season=None):
        c = self.count(("bucket_pos", bucket, pos), exclude_season)
        n = self.count(("bucket", bucket), exclude_season)
        return (c + 1) / (n + len(POSITIONS))

    def league_round_share(self, rnd, pos, exclude_season=None):
        """Share of this exact round's picks the league spends on pos."""
        c = self.count(("round_pos", str(rnd), pos), exclude_season)
        n = self.count(("round", str(rnd)), exclude_season)
        return (c + 0.5) / (n + 0.5 * len(POSITIONS))

    def league_team_share(self, team, exclude_season=None):
        c = self.count(("team", team), exclude_season)
        n = self.count(("all",), exclude_season)
        return (c + 1) / (n + NFL_TEAMS)

    def position_lift(self, owner, bucket, pos, exclude_season=None):
        """log(manager's shrunk share / league share) of pos picks in bucket."""
        league = self.league_position_share(bucket, pos, exclude_season)
        c = self.count(("owner_bucket_pos", owner, bucket, pos), exclude_season)
        n = self.count(("owner_bucket", owner, bucket), exclude_season)
        share = (c + POSITION_PRIOR_PICKS * league) / (n + POSITION_PRIOR_PICKS)
        return math.log(share / league)

    def team_lift(self, owner, team, exclude_season=None):
        """log(manager's shrunk share / league share) of picks from team."""
        if not team:
            return 0.0
        league = self.league_team_share(team, exclude_season)
        c = self.count(("owner_team", owner, team), exclude_season)
        n = self.count(("owner", owner), exclude_season)
        share = (c + TEAM_PRIOR_PICKS * league) / (n + TEAM_PRIOR_PICKS)
        return math.log(share / league)


def candidate_features(player, pick, available_rank, roster, stats, habits, turn):
    season, owner = pick["season"], pick["owner"]
    pos, team = player["position"], player["nfl_team"]
    prior = stats.get((season, player["player_id"]))
    gp = float(prior["gp"]) if prior else 0.0
    count = sum(1 for p in roster if p["position"] == pos)
    gone = player["adp_ppr"] < turn["next_pick"]
    x = {
        "gone_before_next_pick": 1.0 if gone else 0.0,
        "position_run": turn["run"][pos],
        "pos_options_next_pick": turn["options"][pos] - (0 if gone else 1),
        "league_pos_round_share": math.log(
            habits.league_round_share(pick["round"], pos, exclude_season=season)),
        "log_pos_candidates": math.log(turn["candidates"][pos]),
        "adp_value": int(pick["pick_no"]) - player["adp_ppr"],
        "log_available_rank": math.log(available_rank),
        "prior_ppg": float(prior["pts_ppr"]) / gp if prior and gp else 0.0,
        "prior_gp": gp,
        "no_prior_stats": 0.0 if prior else 1.0,
        "pos_count": count,
        "starter_need": 1.0 if count < STARTERS.get(pos, 0) else 0.0,
        "nfl_team_stack": 1.0 if team and any(
            p["nfl_team"] == team for p in roster) else 0.0,
        "owner_pos_round_lift": habits.position_lift(
            owner, round_bucket(pick["round"]), pos, exclude_season=season),
        "owner_nfl_team_lift": habits.team_lift(owner, team, exclude_season=season),
    }
    for p in POSITIONS:
        if p != "RB":
            x[f"pos_{p}"] = 1.0 if pos == p else 0.0
    return [float(x[f]) for f in FEATURES]


def build_choice_sets(picks, adp, stats, habits):
    """One choice set per pick: (owner, candidate feature rows, chosen index)."""
    by_draft = defaultdict(list)
    for p in picks:
        by_draft[p["draft_id"]].append(p)

    choice_sets, meta, skipped = [], [], 0
    for draft_picks in by_draft.values():
        draft_picks.sort(key=lambda p: int(p["pick_no"]))
        # Only positions this draft actually rostered (the league dropped
        # kickers after 2022, so kickers aren't real options in later drafts).
        drafted_positions = {p["position"] for p in draft_picks}
        pool = [r for r in adp[draft_picks[0]["season"]]
                if r["position"] in drafted_positions]
        adp_by_id = {r["player_id"]: r for r in pool}
        taken, rosters = set(), defaultdict(list)
        teams = max(int(p["draft_slot"]) for p in draft_picks)
        own_picks = defaultdict(list)  # owner_id -> their pick numbers, in order
        for p in draft_picks:
            own_picks[p["owner_id"]].append(int(p["pick_no"]))
        positions_taken = []  # position of each pick so far, in order

        for p in draft_picks:
            chosen = adp_by_id.get(p["player_id"])
            pick_no = int(p["pick_no"])
            if chosen:
                available = [r for r in pool if r["player_id"] not in taken]
                candidates = available[:CHOICE_SET_SIZE]
                if chosen not in candidates:
                    candidates.append(chosen)
                rank = {r["player_id"]: i for i, r in enumerate(available, 1)}
                turn = turn_context(own_picks[p["owner_id"]], pick_no, teams,
                                    positions_taken, candidates)
                rows = [candidate_features(c, p, rank[c["player_id"]],
                                           rosters[p["owner"]], stats, habits, turn)
                        for c in candidates]
                choice_sets.append((p["owner"], rows, candidates.index(chosen)))
                meta.append({"season": p["season"], "round": int(p["round"]),
                             "positions": [c["position"] for c in candidates],
                             "position": chosen["position"]})
            else:
                skipped += 1
            taken.add(p["player_id"])
            positions_taken.append(p["position"])
            rosters[p["owner"]].append(
                {"position": p["position"], "nfl_team": p["nfl_team"]})
    return choice_sets, meta, skipped


def turn_context(own_pick_nos, pick_no, teams, positions_taken, candidates):
    """What the picking manager sees from their spot in the draft order."""
    later = [n for n in own_pick_nos if n > pick_no]
    earlier = [n for n in own_pick_nos if n < pick_no]
    next_pick = later[0] if later else pick_no + teams
    # positions taken since their last pick (all earlier picks if this is their first)
    since = positions_taken[earlier[-1]:] if earlier else positions_taken
    options = Counter(c["position"] for c in candidates
                      if c["adp_ppr"] >= next_pick)
    return {"next_pick": next_pick, "run": Counter(since), "options": options,
            "candidates": Counter(c["position"] for c in candidates)}


def scaler(rows):
    """Function that z-scores a feature row using the mean and sd of rows."""
    k = len(rows[0])
    mean = [sum(r[j] for r in rows) / len(rows) for j in range(k)]
    sd = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in rows) / len(rows)) or 1.0
          for j in range(k)]
    return lambda r: [(r[j] - mean[j]) / sd[j] for j in range(k)]


def standardize(choice_sets):
    scale = scaler([x for _, xs, _ in choice_sets for x in xs])
    return [(o, [scale(x) for x in xs], c) for o, xs, c in choice_sets]


def select_features(choice_sets, names):
    idx = [FEATURES.index(n) for n in names]
    return [(o, [[x[j] for j in idx] for x in xs], c) for o, xs, c in choice_sets]


def with_offsets(choice_sets, offsets):
    """Pair each choice set with its per-candidate offsets (None = no offsets)."""
    return zip(choice_sets, offsets or [None] * len(choice_sets))


def probabilities(beta, xs, offsets=None):
    """Logit choice probabilities. Optional offsets are fixed per-candidate
    additions to utility that are not estimated."""
    utils = [sum(b * v for b, v in zip(beta, x)) for x in xs]
    if offsets:
        utils = [u + o for u, o in zip(utils, offsets)]
    top = max(utils)
    exps = [math.exp(u - top) for u in utils]
    total = sum(exps)
    return [e / total for e in exps]


def log_likelihood(beta, choice_sets, offsets=None):
    return sum(math.log(max(probabilities(beta, xs, offs)[c], 1e-300))
               for (_, xs, c), offs in with_offsets(choice_sets, offsets))


def solve(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a_ - f * b_ for a_, b_ in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def information(beta, choice_sets, offsets=None):
    """Gradient and Fisher information of the conditional logit likelihood."""
    k = len(beta)
    grad = [0.0] * k
    info = [[0.0] * k for _ in range(k)]
    for (_, xs, c), offs in with_offsets(choice_sets, offsets):
        p = probabilities(beta, xs, offs)
        mean_x = [sum(pi * x[j] for pi, x in zip(p, xs)) for j in range(k)]
        for j in range(k):
            grad[j] += xs[c][j] - mean_x[j]
        for pi, x in zip(p, xs):
            d = [x[j] - mean_x[j] for j in range(k)]
            for i in range(k):
                for j in range(i, k):
                    info[i][j] += pi * d[i] * d[j]
    for i in range(k):
        for j in range(i):
            info[i][j] = info[j][i]
    return grad, info


def fit(choice_sets, offsets=None):
    """Ridge-penalized conditional logit via Newton's method, halving any step
    that would lower the penalized likelihood so the fit can't diverge."""
    def objective(beta):
        return (log_likelihood(beta, choice_sets, offsets)
                - RIDGE / 2 * sum(b * b for b in beta))

    k = len(choice_sets[0][1][0])
    beta = [0.0] * k
    current = objective(beta)
    for _ in range(NEWTON_ITERS):
        grad, info = information(beta, choice_sets, offsets)
        grad = [g - RIDGE * b for g, b in zip(grad, beta)]
        for i in range(k):
            info[i][i] += RIDGE
        step = solve(info, grad)
        for _ in range(MAX_STEP_HALVINGS):
            trial = [b + s for b, s in zip(beta, step)]
            value = objective(trial)
            if value >= current:
                break
            step = [s / 2 for s in step]
        else:
            break
        beta, current = trial, value
        if max(abs(s) for s in step) < 1e-6:
            break
    return beta


def standard_errors(beta, choice_sets):
    _, info = information(beta, choice_sets)
    k = len(beta)
    return [math.sqrt(max(solve(info, [1.0 if i == j else 0.0 for i in range(k)])[j], 0.0))
            for j in range(k)]


def top_n_accuracy(beta, choice_sets, n):
    hits = 0
    for _, xs, c in choice_sets:
        p = probabilities(beta, xs)
        hits += c in sorted(range(len(p)), key=lambda i: -p[i])[:n]
    return hits / len(choice_sets)


def permutation_importance(beta, choice_sets, base_ll):
    """Drop in log-likelihood per pick when one trait is shuffled."""
    rng = random.Random(SEED)
    drops = []
    for j in range(len(beta)):
        values = [x[j] for _, xs, _ in choice_sets for x in xs]
        total = 0.0
        for _ in range(PERMUTATION_REPEATS):
            rng.shuffle(values)
            it = iter(values)
            shuffled = [(o, [x[:j] + [next(it)] + x[j + 1:] for x in xs], c)
                        for o, xs, c in choice_sets]
            total += base_ll - log_likelihood(beta, shuffled)
        drops.append(total / PERMUTATION_REPEATS / len(choice_sets))
    return drops


def is_early(m):
    return m["round"] <= FOCUS_ROUNDS


def out_of_sample_probabilities(choice_sets, meta, keep=lambda m: True):
    """Each kept pick's candidate probabilities, from a model fit only on the
    other seasons' kept picks. Picks not kept get None."""
    probs = [None] * len(choice_sets)
    for season in sorted({m["season"] for m in meta}):
        train = [cs for cs, m in zip(choice_sets, meta)
                 if keep(m) and m["season"] != season]
        beta = fit(train)
        for i, (cs, m) in enumerate(zip(choice_sets, meta)):
            if keep(m) and m["season"] == season:
                probs[i] = probabilities(beta, cs[1])
    return probs


def out_of_sample_predictions(choice_sets, meta, keep=lambda m: True):
    """For each kept pick, (rank of the real pick in the model's list, log
    probability the model gave it). Picks not kept get None."""
    return predictions_from(choice_sets, out_of_sample_probabilities(choice_sets, meta, keep))


def predictions_from(choice_sets, probs):
    preds = []
    for (_, _, c), p in zip(choice_sets, probs):
        preds.append(None if p is None else
                     (1 + sum(pi > p[c] for pi in p), math.log(max(p[c], MIN_PROBABILITY))))
    return preds


def position_probabilities(probs, positions):
    """Collapse candidate probabilities into probabilities per position."""
    by_pos = Counter()
    for p, pos in zip(probs, positions):
        by_pos[pos] += p
    return by_pos


def round_position_rates(meta):
    """Leave-one-season-out share of each position taken in each round."""
    counts = Counter((m["season"], m["round"], m["position"]) for m in meta)
    totals = Counter((m["season"], m["round"]) for m in meta)
    seasons = {m["season"] for m in meta}
    rates = {}
    for m in meta:
        key = (m["season"], m["round"])
        if key in rates:
            continue
        others = [s for s in seasons if s != m["season"]]
        n = sum(totals[(s, m["round"])] for s in others)
        rates[key] = Counter({
            pos: (sum(counts[(s, m["round"], pos)] for s in others) + 0.5)
                 / (n + 0.5 * len(POSITIONS))
            for pos in POSITIONS})
    return rates


def position_accuracy_rows(meta, probs_by_model):
    """How often the predicted position is right, by stage of the draft."""
    rates = round_position_rates(meta)
    groups = [(f"{r}", lambda m, r=r: m["round"] == r) for r in range(1, FOCUS_ROUNDS + 1)]
    groups += [(f"1-{FOCUS_ROUNDS}", is_early),
               ("6-10", lambda m: 6 <= m["round"] <= 10),
               ("11+", lambda m: m["round"] >= 11),
               ("all", lambda m: True)]
    rows = []
    for label, keep in groups:
        idx = [i for i, m in enumerate(meta) if keep(m)]
        n = len(idx)
        row = {"rounds": label, "picks": n,
               "best_available_pos": round(
                   sum(meta[i]["positions"][0] == meta[i]["position"] for i in idx) / n, 3)}
        round_rate = [rates[(meta[i]["season"], meta[i]["round"])] for i in idx]
        row["round_history_top1"] = round(sum(
            max(r, key=r.get) == meta[i]["position"] for r, i in zip(round_rate, idx)) / n, 3)
        row["round_history_log_score"] = round(sum(
            math.log(r[meta[i]["position"]]) for r, i in zip(round_rate, idx)) / n, 3)
        for name, probs in probs_by_model.items():
            if any(probs[i] is None for i in idx):
                row.update({f"{name}_{s}": "" for s in ("top1", "top2", "log_score")})
                continue
            pos_probs = [position_probabilities(probs[i], meta[i]["positions"]) for i in idx]
            actual = [meta[i]["position"] for i in idx]
            row[f"{name}_top1"] = round(sum(
                max(pp, key=pp.get) == a for pp, a in zip(pos_probs, actual)) / n, 3)
            row[f"{name}_top2"] = round(sum(
                a in sorted(pp, key=pp.get, reverse=True)[:2]
                for pp, a in zip(pos_probs, actual)) / n, 3)
            row[f"{name}_log_score"] = round(sum(
                math.log(max(pp[a], MIN_PROBABILITY)) for pp, a in zip(pos_probs, actual)) / n, 3)
        rows.append(row)
    return rows


def round_accuracy_rows(raw_sets, meta, preds_by_model):
    groups = [(f"{r}", lambda m, r=r: m["round"] == r) for r in range(1, FOCUS_ROUNDS + 1)]
    groups += [(f"1-{FOCUS_ROUNDS}", is_early),
               (f"{FOCUS_ROUNDS + 1}+", lambda m: not is_early(m)),
               ("6-10", lambda m: 6 <= m["round"] <= 10),
               ("11+", lambda m: m["round"] >= 11),
               ("all", lambda m: True)]
    rows = []
    for label, keep in groups:
        idx = [i for i, m in enumerate(meta) if keep(m)]
        row = {"rounds": label, "picks": len(idx),
               "best_available_top1": round(
                   sum(raw_sets[i][2] == 0 for i in idx) / len(idx), 3)}
        for name, preds in preds_by_model.items():
            got = [preds[i] for i in idx]
            if None in got:
                row.update({f"{name}_{s}": "" for s in ("top1", "top3", "log_score")})
                continue
            row[f"{name}_top1"] = round(sum(r == 1 for r, _ in got) / len(got), 3)
            row[f"{name}_top3"] = round(sum(r <= 3 for r, _ in got) / len(got), 3)
            # mean log probability given to the real pick; closer to 0 is better
            row[f"{name}_log_score"] = round(sum(lp for _, lp in got) / len(got), 3)
        rows.append(row)
    return rows


def importance_rows(choice_sets, names):
    beta = fit(choice_sets)
    ses = standard_errors(beta, choice_sets)
    drops = permutation_importance(beta, choice_sets, log_likelihood(beta, choice_sets))
    rows = [{
        "trait": f,
        "coef_per_sd": round(b, 4),
        "std_error": round(se, 4),
        "z": round(b / se, 2) if se else "",
        "odds_ratio_per_sd": round(math.exp(b), 3),
        "importance_ll_drop_per_pick": round(d, 4),
    } for f, b, se, d in zip(names, beta, ses, drops)]
    rows.sort(key=lambda r: -r["importance_ll_drop_per_pick"])
    return rows


def print_importance(title, rows):
    print(f"\n{title}")
    print(f"{'trait':<22}{'coef/sd':>9}{'z':>8}{'importance':>12}")
    for r in rows:
        print(f"{r['trait']:<22}{r['coef_per_sd']:>9.3f}{r['z']:>8}"
              f"{r['importance_ll_drop_per_pick']:>12.4f}")


def position_pace_rows(meta, probs_by_model, max_round=16):
    """Cumulative players taken per position by the end of each round, actual
    vs. expected (summed position probabilities), averaged over seasons.
    Each pick is predicted knowing the picks before it, not simulated."""
    rates = round_position_rates(meta)
    seasons = sorted({m["season"] for m in meta})
    sums = defaultdict(float)  # (source, round, pos) -> total over seasons
    for season in seasons:
        running = defaultdict(float)
        idx = [i for i, m in enumerate(meta) if m["season"] == season]
        for rnd in range(1, max_round + 1):
            for i in (i for i in idx if meta[i]["round"] == rnd):
                m = meta[i]
                running[("actual", m["position"])] += 1
                running[("best_available", m["positions"][0])] += 1
                for pos, p in rates[(season, rnd)].items():
                    running[("round_history", pos)] += p
                for name, probs in probs_by_model.items():
                    for pos, p in position_probabilities(probs[i], m["positions"]).items():
                        running[(name, pos)] += p
            for (source, pos), v in running.items():
                sums[(source, rnd, pos)] += v
    sources = ["actual", "best_available", "round_history", *probs_by_model]
    rows = []
    for rnd in range(1, max_round + 1):
        for pos in POSITIONS:
            if pos == "K":
                continue
            rows.append({"round": rnd, "position": pos,
                         **{s: round(sums[(s, rnd, pos)] / len(seasons), 2) for s in sources}})
    return rows


def poisson_upper_tail(observed, expected):
    """P(X >= observed) for X ~ Poisson(expected)."""
    term = cdf = math.exp(-expected)
    for k in range(1, observed):
        term *= expected / k
        cdf += term
    return max(0.0, 1.0 - cdf) if observed > 0 else 1.0


def position_tendency_rows(picks, habits):
    owners = sorted({p["owner"] for p in picks})
    rows = []
    for owner in owners:
        for _, _, bucket in ROUND_BUCKETS:
            n = habits.count(("owner_bucket", owner, bucket))
            for pos in POSITIONS:
                c = habits.count(("owner_bucket_pos", owner, bucket, pos))
                league = habits.league_position_share(bucket, pos)
                rows.append({
                    "owner": owner, "rounds": bucket, "position": pos,
                    "picks": c, "owner_share": round(c / n, 3) if n else "",
                    "league_share": round(league, 3),
                    "lift": round(math.exp(habits.position_lift(owner, bucket, pos)), 2),
                })
    return rows


def team_affinity_rows(picks, habits):
    owners = sorted({p["owner"] for p in picks})
    teams = sorted({p["nfl_team"] for p in picks if p["nfl_team"]})
    rows = []
    for owner in owners:
        n = habits.count(("owner", owner))
        for team in teams:
            c = habits.count(("owner_team", owner, team))
            expected = n * habits.league_team_share(team)
            if c < 2:
                continue
            rows.append({
                "owner": owner, "nfl_team": team, "picks": c,
                "expected": round(expected, 2), "ratio": round(c / expected, 2),
                "lift": round(math.exp(habits.team_lift(owner, team)), 2),
                "p_value": round(poisson_upper_tail(c, expected), 4),
            })
    rows.sort(key=lambda r: r["p_value"])
    return rows


def main():
    picks, adp, stats = load_inputs()
    habits = Tendencies(picks)
    raw_sets, meta, skipped = build_choice_sets(picks, adp, stats, habits)
    choice_sets = standardize(raw_sets)
    print(f"{len(choice_sets)} picks modeled ({skipped} skipped: no Sleeper ADP)")

    null_ll = sum(-math.log(len(xs)) for _, xs, _ in choice_sets)
    best_available = sum(c == 0 for _, _, c in raw_sets) / len(raw_sets)
    print(f"Baseline 'take best available by ADP': top-1 {best_available:.1%}")
    models = {"player traits only": PLAYER_FEATURES,
              "+ draft position": BASE_FEATURES,
              "+ league pos prefs": PREF_FEATURES,
              "+ manager habits": FEATURES}
    for label, names in models.items():
        sets = select_features(choice_sets, names)
        beta = fit(sets)
        ll = log_likelihood(beta, sets)
        print(f"{label:<20} pseudo-R2 {1 - ll / null_ll:.3f}  "
              f"top-1 {top_n_accuracy(beta, sets, 1):.1%}  "
              f"top-3 {top_n_accuracy(beta, sets, 3):.1%}")

    draft_sets = select_features(choice_sets, BASE_FEATURES)
    pref_sets = select_features(choice_sets, PREF_FEATURES)
    probs_by_model = {
        "draftpos": out_of_sample_probabilities(draft_sets, meta),
        "prefs": out_of_sample_probabilities(pref_sets, meta),
        "habits": out_of_sample_probabilities(choice_sets, meta),
    }
    preds_by_model = {
        "all_rounds_draftpos": predictions_from(draft_sets, probs_by_model["draftpos"]),
        "all_rounds_prefs": predictions_from(pref_sets, probs_by_model["prefs"]),
        "early_draftpos": out_of_sample_predictions(draft_sets, meta, is_early),
        "early_prefs": out_of_sample_predictions(pref_sets, meta, is_early),
    }

    pace = position_pace_rows(meta, probs_by_model)
    write_csv(POSITION_PACE_CSV, list(pace[0]), pace)
    print("\nPosition pace: mean miss in cumulative players per position")
    for source in ["round_history", *probs_by_model]:
        for label, keep in [("all rounds", lambda r: True),
                            ("rounds 6-10", lambda r: 6 <= r["round"] <= 10)]:
            misses = [abs(r[source] - r["actual"]) for r in pace if keep(r)]
            print(f"  {source:<14} {label:<12} {sum(misses) / len(misses):.2f}")
    pos_rows = position_accuracy_rows(meta, probs_by_model)
    write_csv(POSITION_ACCURACY_CSV, list(pos_rows[0]), pos_rows)
    print("\nPosition prediction, out of sample; model cells are top-1 / top-2 / log score")
    print(f"{'rounds':<7}{'picks':>6}{'best avail':>11}{'round hist':>16}" +
          "".join(f"{name:>24}" for name in probs_by_model))
    for r in pos_rows:
        cells = "".join(
            f"{r[f'{n}_top1']:>9.1%} {r[f'{n}_top2']:>6.1%} {r[f'{n}_log_score']:>6.2f}"
            for n in probs_by_model)
        print(f"{r['rounds']:<7}{r['picks']:>6}{r['best_available_pos']:>11.1%}"
              f"{r['round_history_top1']:>9.1%} {r['round_history_log_score']:>6.2f}{cells}")
    accuracy = round_accuracy_rows(raw_sets, meta, preds_by_model)
    write_csv(ROUND_ACCURACY_CSV, list(accuracy[0]), accuracy)
    print("\nOut-of-sample accuracy (each season predicted from the others);"
          " cells are top-1 / top-3 / log score")
    print(f"{'rounds':<7}{'picks':>6}{'best':>7}" +
          "".join(f"{name:>24}" for name in preds_by_model))
    for r in accuracy:
        cells = "".join(
            f"{'-':>24}" if r[f'{n}_top1'] == "" else
            f"{r[f'{n}_top1']:>9.1%} {r[f'{n}_top3']:>6.1%} {r[f'{n}_log_score']:>6.2f}"
            for n in preds_by_model)
        print(f"{r['rounds']:<7}{r['picks']:>6}{r['best_available_top1']:>7.1%}{cells}")

    rows = importance_rows(choice_sets, FEATURES)
    write_csv(IMPORTANCE_CSV, list(rows[0]), rows)
    print_importance("All-rounds model traits:", rows)

    early_sets = [cs for cs, m in zip(choice_sets, meta) if is_early(m)]
    early_rows = importance_rows(early_sets, FEATURES)
    write_csv(EARLY_IMPORTANCE_CSV, list(early_rows[0]), early_rows)
    print_importance(f"Rounds 1-{FOCUS_ROUNDS} model traits:", early_rows)

    write_csv(POSITION_TENDENCIES_CSV,
              ["owner", "rounds", "position", "picks", "owner_share",
               "league_share", "lift"],
              position_tendency_rows(picks, habits))
    affinity = team_affinity_rows(picks, habits)
    write_csv(TEAM_AFFINITY_CSV,
              ["owner", "nfl_team", "picks", "expected", "ratio", "lift", "p_value"],
              affinity)
    print("\nStrongest favorite-team signals:")
    for r in affinity[:8]:
        print(f"  {r['owner']:<14} {r['nfl_team']:<4} {r['picks']} picks "
              f"vs {r['expected']} expected (p={r['p_value']})")


if __name__ == "__main__":
    main()
