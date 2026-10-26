"""Two-step pick model: first which position, then which player at it.

P(player) = P(position) x P(player | position)

Step 1 chooses among every position still on the board, using how often the
league takes that position in this round, the manager's roster, position runs
and how good / how scarce the best remaining players at each position are.
Step 2 chooses among the best remaining players at one position, using player
traits only. Splitting the two keeps position totals in line with how the
league actually drafts, while still ranking individual players.

Each season is predicted by a model fit only on the other seasons.

Outputs (in results/):
  - two_stage_accuracy.csv: player and position accuracy by stage of draft
  - two_stage_pace.csv: cumulative players per position by end of each round,
    actual vs. expected

Run fetch_data.py first to build the input CSVs.
"""

import math
from collections import defaultdict

import pick_model as pm

ACCURACY_CSV = pm.RESULTS_DIR / "two_stage_accuracy.csv"
PACE_CSV = pm.RESULTS_DIR / "two_stage_pace.csv"
PLAYERS_PER_POSITION = 10  # step-2 candidates: best remaining at each position
MAX_ROUND = 16

# Step 1 starts every position at the league's share of this round's picks
# (from other seasons) and these traits adjust up or down from there. With
# nothing else known, the model reproduces the league's round-by-round mix.
POSITION_FEATURES = [
    # manager's players already at the position; each position its own trait,
    # since a second RB and a second QB mean very different things
    "rb_count", "wr_count", "qb_count", "te_count",
    "starter_need",            # manager still has an open starting slot there
    "position_run",            # picks at the position since the manager last picked
    "best_log_rank",           # log overall board rank of the best one left
    "best_adp_value",          # pick number minus ADP of the best one left (+ = fallen)
    "best_gone",               # best one left won't last to the manager's next pick
    # log of players at the position ADP says go before next pick; a log so the
    # naturally bigger WR pool doesn't read as proportionally more pressure
    "log_supply_before_next",
    # Roster limits differ by position (one DEF is enough, a second WR is
    # routine), so "starter already filled" is its own trait per position.
    "qb_filled", "te_filled", "def_filled",
    "pos_QB", "pos_WR", "pos_TE", "pos_DEF",  # vs. RB
]
POSITION_HABIT_FEATURES = ["owner_pos_round_lift"]

PLAYER_FEATURES = [
    "adp_value", "log_available_rank", "prior_ppg", "prior_gp", "no_prior_stats",
    "nfl_team_stack", "gone_before_next_pick",
]
PLAYER_HABIT_FEATURES = ["owner_nfl_team_lift"]

VARIANTS = {
    "two_stage": (POSITION_FEATURES, PLAYER_FEATURES),
    "two_stage_habits": (POSITION_FEATURES + POSITION_HABIT_FEATURES,
                         PLAYER_FEATURES + PLAYER_HABIT_FEATURES),
}
ALL_POSITION_FEATURES = POSITION_FEATURES + POSITION_HABIT_FEATURES
ALL_PLAYER_FEATURES = PLAYER_FEATURES + PLAYER_HABIT_FEATURES


def position_features(pos, players, pick, roster, habits, turn, rank):
    """Features of one position option; players are those left at it, by ADP."""
    best = players[0]
    count = sum(1 for p in roster if p["position"] == pos)
    season, rnd = pick["season"], pick["round"]
    x = {
        "starter_need": 1.0 if count < pm.STARTERS.get(pos, 0) else 0.0,
        "position_run": turn["run"][pos],
        "best_log_rank": math.log(rank[best["player_id"]]),
        "best_adp_value": int(pick["pick_no"]) - best["adp_ppr"],
        "best_gone": 1.0 if best["adp_ppr"] < turn["next_pick"] else 0.0,
        "log_supply_before_next": math.log1p(
            sum(1 for p in players if p["adp_ppr"] < turn["next_pick"])),
        "owner_pos_round_lift": habits.position_lift(
            pick["owner"], pm.round_bucket(rnd), pos, exclude_season=season),
    }
    filled = count >= pm.STARTERS.get(pos, 0)
    for p in ("QB", "TE", "DEF"):
        x[f"{p.lower()}_filled"] = 1.0 if pos == p and filled else 0.0
    for p in ("QB", "WR", "TE", "DEF"):
        x[f"pos_{p}"] = 1.0 if pos == p else 0.0
    for p in ("RB", "WR", "QB", "TE"):
        x[f"{p.lower()}_count"] = float(count) if pos == p else 0.0
    return [float(x[f]) for f in ALL_POSITION_FEATURES]


def build_picks(picks, adp, stats, habits):
    """Replay each draft; one record per modeled pick with both steps' options."""
    player_idx = [pm.FEATURES.index(f) for f in ALL_PLAYER_FEATURES]
    by_draft = defaultdict(list)
    for p in picks:
        by_draft[p["draft_id"]].append(p)

    records, skipped = [], 0
    for draft_picks in by_draft.values():
        draft_picks.sort(key=lambda p: int(p["pick_no"]))
        drafted_positions = {p["position"] for p in draft_picks}
        pool = [r for r in adp[draft_picks[0]["season"]]
                if r["position"] in drafted_positions]
        adp_by_id = {r["player_id"]: r for r in pool}
        teams = max(int(p["draft_slot"]) for p in draft_picks)
        own_picks = defaultdict(list)
        for p in draft_picks:
            own_picks[p["owner_id"]].append(int(p["pick_no"]))
        taken, rosters, positions_taken = set(), defaultdict(list), []

        for p in draft_picks:
            chosen = adp_by_id.get(p["player_id"])
            pick_no = int(p["pick_no"])
            if chosen:
                available = [r for r in pool if r["player_id"] not in taken]
                rank = {r["player_id"]: i for i, r in enumerate(available, 1)}
                by_pos = defaultdict(list)
                for r in available:
                    by_pos[r["position"]].append(r)
                turn = pm.turn_context(own_picks[p["owner_id"]], pick_no, teams,
                                        positions_taken, available[:pm.CHOICE_SET_SIZE])
                roster = rosters[p["owner"]]
                positions = sorted(by_pos)
                # candidate_features takes the log of the per-position
                # candidate count. It feeds no step-2 feature, so it only needs
                # to be positive for positions outside the top CHOICE_SET_SIZE.
                for pos in positions:
                    turn["candidates"][pos] = max(1, turn["candidates"][pos])
                options = {}
                for pos in positions:
                    players = by_pos[pos][:PLAYERS_PER_POSITION]
                    if pos == chosen["position"] and chosen not in players:
                        players = players + [chosen]
                    rows = [pm.candidate_features(c, p, rank[c["player_id"]], roster,
                                                   stats, habits, turn) for c in players]
                    options[pos] = {
                        "players": [c["player_id"] for c in players],
                        "rows": [[r[j] for j in player_idx] for r in rows],
                    }
                records.append({
                    "season": p["season"], "round": int(p["round"]), "owner": p["owner"],
                    "positions": positions,
                    "pos_rows": [position_features(pos, by_pos[pos], p, roster, habits,
                                                   turn, rank) for pos in positions],
                    "pos_offsets": [math.log(habits.league_round_share(
                        p["round"], pos, exclude_season=p["season"])) for pos in positions],
                    "chosen_pos": chosen["position"],
                    "chosen_id": chosen["player_id"],
                    "options": options,
                })
            else:
                skipped += 1
            taken.add(p["player_id"])
            positions_taken.append(p["position"])
            rosters[p["owner"]].append({"position": p["position"], "nfl_team": p["nfl_team"]})
    return records, skipped


def standardize(records):
    pos_scale = pm.scaler([r for rec in records for r in rec["pos_rows"]])
    player_scale = pm.scaler([r for rec in records for o in rec["options"].values()
                              for r in o["rows"]])
    for rec in records:
        rec["pos_rows"] = [pos_scale(r) for r in rec["pos_rows"]]
        for o in rec["options"].values():
            o["rows"] = [player_scale(r) for r in o["rows"]]


def subset(rows, idx):
    return [[r[j] for j in idx] for r in rows]


def fit_variant(train, pos_idx, player_idx):
    pos_sets = [(r["owner"], subset(r["pos_rows"], pos_idx),
                 r["positions"].index(r["chosen_pos"])) for r in train]
    pos_offsets = [r["pos_offsets"] for r in train]
    player_sets = []
    for r in train:
        o = r["options"][r["chosen_pos"]]
        if len(o["players"]) > 1:
            player_sets.append((r["owner"], subset(o["rows"], player_idx),
                                o["players"].index(r["chosen_id"])))
    return pm.fit(pos_sets, pos_offsets), pm.fit(player_sets)


def predict(rec, pos_beta, player_beta, pos_idx, player_idx):
    """Position probabilities and every option player's probability."""
    pos_p = dict(zip(rec["positions"], pm.probabilities(
        pos_beta, subset(rec["pos_rows"], pos_idx), rec["pos_offsets"])))
    player_p = {}
    for pos, o in rec["options"].items():
        within = pm.probabilities(player_beta, subset(o["rows"], player_idx))
        for pid, q in zip(o["players"], within):
            player_p[pid] = pos_p[pos] * q
    return pos_p, player_p


def out_of_sample(records, pos_features, player_features):
    pos_idx = [ALL_POSITION_FEATURES.index(f) for f in pos_features]
    player_idx = [ALL_PLAYER_FEATURES.index(f) for f in player_features]
    preds = [None] * len(records)
    for season in sorted({r["season"] for r in records}):
        train = [r for r in records if r["season"] != season]
        pos_beta, player_beta = fit_variant(train, pos_idx, player_idx)
        for i, rec in enumerate(records):
            if rec["season"] == season:
                preds[i] = predict(rec, pos_beta, player_beta, pos_idx, player_idx)
    return preds


GROUPS = ([(str(r), lambda rec, r=r: rec["round"] == r) for r in range(1, 6)] +
          [("1-5", lambda rec: rec["round"] <= 5),
           ("6-10", lambda rec: 6 <= rec["round"] <= 10),
           ("11+", lambda rec: rec["round"] >= 11),
           ("all", lambda rec: True)])


def accuracy_rows(records, preds_by_variant):
    rows = []
    for label, keep in GROUPS:
        idx = [i for i, rec in enumerate(records) if keep(rec)]
        n = len(idx)
        row = {"rounds": label, "picks": n}
        for name, preds in preds_by_variant.items():
            ranks, logs, pos_hit, pos_top2, pos_log = [], [], 0, 0, 0.0
            for i in idx:
                rec = records[i]
                pos_p, player_p = preds[i]
                pc = player_p[rec["chosen_id"]]
                ranks.append(1 + sum(q > pc for q in player_p.values()))
                logs.append(math.log(max(pc, pm.MIN_PROBABILITY)))
                order = sorted(pos_p, key=pos_p.get, reverse=True)
                pos_hit += order[0] == rec["chosen_pos"]
                pos_top2 += rec["chosen_pos"] in order[:2]
                pos_log += math.log(max(pos_p[rec["chosen_pos"]], pm.MIN_PROBABILITY))
            row.update({
                f"{name}_player_top1": round(sum(r == 1 for r in ranks) / n, 3),
                f"{name}_player_top3": round(sum(r <= 3 for r in ranks) / n, 3),
                f"{name}_player_log_score": round(sum(logs) / n, 3),
                f"{name}_pos_top1": round(pos_hit / n, 3),
                f"{name}_pos_top2": round(pos_top2 / n, 3),
                f"{name}_pos_log_score": round(pos_log / n, 3),
            })
        rows.append(row)
    return rows


def pace_rows(records, preds_by_variant):
    seasons = sorted({r["season"] for r in records})
    sums = defaultdict(float)
    for season in seasons:
        running = defaultdict(float)
        idx = [i for i, rec in enumerate(records) if rec["season"] == season]
        for rnd in range(1, MAX_ROUND + 1):
            for i in (i for i in idx if records[i]["round"] == rnd):
                running[("actual", records[i]["chosen_pos"])] += 1
                for name, preds in preds_by_variant.items():
                    for pos, q in preds[i][0].items():
                        running[(name, pos)] += q
            for (source, pos), v in running.items():
                sums[(source, rnd, pos)] += v
    rows = []
    for rnd in range(1, MAX_ROUND + 1):
        for pos in pm.POSITIONS:
            if pos == "K":
                continue
            rows.append({"round": rnd, "position": pos,
                         **{s: round(sums[(s, rnd, pos)] / len(seasons), 2)
                            for s in ["actual", *preds_by_variant]}})
    return rows


def main():
    picks, adp, stats = pm.load_inputs()
    habits = pm.Tendencies(picks)
    records, skipped = build_picks(picks, adp, stats, habits)
    standardize(records)
    print(f"{len(records)} picks modeled ({skipped} skipped: no Sleeper ADP)")

    preds_by_variant = {name: out_of_sample(records, pf, plf)
                        for name, (pf, plf) in VARIANTS.items()}

    acc = accuracy_rows(records, preds_by_variant)
    pm.write_csv(ACCURACY_CSV, list(acc[0]), acc)
    print("\nOut of sample; cells are player top-1 / top-3 | position top-1 / top-2")
    print(f"{'rounds':<7}{'picks':>6}" + "".join(f"{n:>34}" for n in preds_by_variant))
    for r in acc:
        cells = "".join(
            f"{r[f'{n}_player_top1']:>11.1%} {r[f'{n}_player_top3']:>6.1%} |"
            f"{r[f'{n}_pos_top1']:>7.1%} {r[f'{n}_pos_top2']:>6.1%}"
            for n in preds_by_variant)
        print(f"{r['rounds']:<7}{r['picks']:>6}{cells}")

    pace = pace_rows(records, preds_by_variant)
    pm.write_csv(PACE_CSV, list(pace[0]), pace)
    print("\nPosition pace: mean miss in cumulative players per position")
    for name in preds_by_variant:
        for label, keep in [("all rounds", lambda r: True),
                            ("rounds 6-10", lambda r: 6 <= r["round"] <= 10)]:
            misses = [abs(r[name] - r["actual"]) for r in pace if keep(r)]
            print(f"  {name:<18} {label:<12} {sum(misses) / len(misses):.2f}")


if __name__ == "__main__":
    main()
