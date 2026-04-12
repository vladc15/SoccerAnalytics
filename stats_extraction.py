import json
import os
import zipfile
import pandas as pd
import numpy as np
from collections import defaultdict

TEAM_MATCHES_CSV = "Olympiacos Piraeus"
TEAM_STATSBOMB = "Olympiacos"
ZIP_PATHS = [
    "data/statsbomb/league_phase.zip",
    "data/statsbomb/playoffs.zip",
]
MATCHES_CSV = "data/matches.csv"

def load_events(match_ids, zip_paths):
    """Load all events for a list of match IDs from the zip archive.
    Returns a flat list of (match_id, event) tuples."""
    records = []

    for one_zip_path in zip_paths:
        with zipfile.ZipFile(one_zip_path, "r") as zf:
            names = zf.namelist()
            for mid in match_ids:
                candidates = [n for n in names if n.endswith(f"{mid}.json")]
                if not candidates:
                    continue
                with zf.open(candidates[0]) as f:
                    for ev in json.load(f):
                        records.append((mid, ev))
    return records

def team_events(records, team):
    """Filter events belonging to a specific team."""
    return [(mid, ev) for mid, ev in records if ev.get("team", {}).get("name") == team]


def load_match_events(match_ids, zip_paths):
    """Load events keyed by match_id to preserve within-match sequencing."""
    events_by_match = {}
    for one_zip_path in zip_paths:
        with zipfile.ZipFile(one_zip_path, "r") as zf:
            names = set(zf.namelist())
            for mid in match_ids:
                name = f"{mid}.json"
                if name not in names:
                    continue
                with zf.open(name) as f:
                    events_by_match[mid] = json.load(f)
    return events_by_match


def event_type_name(ev):
    return ev.get("type", {}).get("name") or "Unknown"


def player_name(ev):
    return ev.get("player", {}).get("name")


def event_descriptor(ev):
    """Compact label for the event outcome used in CSV detail columns."""
    etype = event_type_name(ev)

    if etype == "Pass":
        outcome = ev.get("pass", {}).get("outcome", {}).get("name")
        return f"Pass - {outcome or 'Complete'}"
    if etype == "Dribble":
        outcome = ev.get("dribble", {}).get("outcome", {}).get("name")
        return f"Dribble - {outcome or 'Unknown'}"
    if etype == "Duel":
        outcome = ev.get("duel", {}).get("outcome", {}).get("name")
        return f"Duel - {outcome or 'Unknown'}"
    if etype == "Shot":
        outcome = ev.get("shot", {}).get("outcome", {}).get("name")
        return f"Shot - {outcome or 'Unknown'}"
    if etype == "Goal Keeper":
        outcome = ev.get("goalkeeper", {}).get("outcome", {}).get("name")
        return f"Goal Keeper - {outcome or 'Unknown'}"
    if etype == "50/50":
        outcome = ev.get("50_50", {}).get("outcome", {}).get("name")
        return f"50/50 - {outcome or 'Unknown'}"
    if etype == "Bad Behaviour":
        card = ev.get("bad_behaviour", {}).get("card", {}).get("name")
        return f"Bad Behaviour - {card or 'Unknown'}"
    return etype


def classify_pressure_episode(last_ev):
    """
    Classify the final event of a consolidated under-pressure episode.

    Returns:
      - pass_success
      - shot
      - dribble_escape
      - duel_escape
      - other_success
      - primary_mistake
      - other_mistake
    """
    etype = event_type_name(last_ev)
    descriptor = event_descriptor(last_ev)

    if etype == "Pass":
        if last_ev.get("pass", {}).get("outcome"):
            return "primary_mistake", descriptor
        return "pass_success", descriptor

    if etype == "Shot":
        return "shot", descriptor

    if etype == "Miscontrol":
        return "primary_mistake", descriptor

    if etype == "Dispossessed":
        return "primary_mistake", descriptor

    if etype == "Dribble":
        outcome = last_ev.get("dribble", {}).get("outcome", {}).get("name")
        if outcome == "Complete":
            return "dribble_escape", descriptor
        return "other_mistake", descriptor

    if etype == "Duel":
        outcome = last_ev.get("duel", {}).get("outcome", {}).get("name")
        if outcome in {"Lost In Play", "Lost Out"}:
            return "primary_mistake", descriptor
        if outcome in {"Won", "Success In Play", "Success Out"}:
            return "duel_escape", descriptor
        return "other_success", descriptor

    if etype == "50/50":
        outcome = last_ev.get("50_50", {}).get("outcome", {}).get("name")
        if outcome and "Lost" in outcome:
            return "other_mistake", descriptor
        return "other_success", descriptor

    # Carry / Clearance / Foul Won / Ball Recovery / etc.
    return "other_success", descriptor


def first_possession_regain_index(events, start_idx, team):
    """
    Return the first global event index where team is back in possession
    after a mistake sequence. Returns len(events) if possession is not regained.
    """
    for idx in range(start_idx, len(events)):
        if events[idx].get("possession_team", {}).get("name") == team:
            return idx
    return len(events)


def scan_mistake_fallout(events, start_idx, regain_idx, team, player):
    """
    Inspect events after a mistake until the team regains possession.

    Tracks whether the same player commits a foul or gets booked, and whether
    the team concedes during the opponent possession(s).
    """
    fallout = {
        "fouled_after_mistake": False,
        "booked_after_mistake": False,
        "conceded_after_mistake": False,
    }

    for ev in events[start_idx:regain_idx]:
        etype = event_type_name(ev)
        ev_player = player_name(ev)

        if etype == "Foul Committed" and ev_player == player:
            fallout["fouled_after_mistake"] = True
            card = ev.get("foul_committed", {}).get("card", {}).get("name")
            if card:
                fallout["booked_after_mistake"] = True

        if etype == "Bad Behaviour" and ev_player == player:
            card = ev.get("bad_behaviour", {}).get("card", {}).get("name")
            if card:
                fallout["booked_after_mistake"] = True

        if (
            etype == "Shot"
            and ev.get("team", {}).get("name") != team
            and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal"
        ):
            fallout["conceded_after_mistake"] = True

        if etype == "Own Goal Against" and ev.get("team", {}).get("name") == team:
            fallout["conceded_after_mistake"] = True

    return fallout


def goal_or_penalty_conceded_before_regain(events, start_idx, regain_idx, team):
    """
    Check whether the team concedes a goal or penalty before it regains possession.
    """
    goal_conceded = False
    penalty_conceded = False

    for ev in events[start_idx:regain_idx]:
        etype = event_type_name(ev)

        if (
            etype == "Shot"
            and ev.get("team", {}).get("name") != team
            and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal"
        ):
            goal_conceded = True

        if etype == "Own Goal Against" and ev.get("team", {}).get("name") == team:
            goal_conceded = True

        if (
            etype == "Foul Committed"
            and ev.get("team", {}).get("name") == team
            and ev.get("foul_committed", {}).get("penalty")
        ):
            penalty_conceded = True

    return {
        "goal_conceded": goal_conceded,
        "penalty_conceded": penalty_conceded,
        "goal_or_penalty_conceded": goal_conceded or penalty_conceded,
    }


def summarize_counter(counter_obj):
    if not counter_obj:
        return ""
    return "; ".join(
        f"{label} ({count})"
        for label, count in sorted(counter_obj.items(), key=lambda item: (-item[1], item[0]))
    )


def iter_pressure_episodes(events, team):
    """
    Yield consolidated under-pressure episodes for a team.

    An episode is a consecutive sequence of team events by the same player in the
    same possession while `under_pressure` stays true.
    """
    team_ev = [ev for ev in events if ev.get("team", {}).get("name") == team and player_name(ev)]
    global_index_lookup = {ev.get("index"): idx for idx, ev in enumerate(events)}

    i = 0
    while i < len(team_ev):
        start_ev = team_ev[i]
        if not start_ev.get("under_pressure"):
            i += 1
            continue

        player = player_name(start_ev)
        possession = start_ev.get("possession")

        j = i + 1
        while j < len(team_ev):
            next_ev = team_ev[j]
            if player_name(next_ev) != player:
                break
            if next_ev.get("possession") != possession:
                break
            if not next_ev.get("under_pressure"):
                break
            j += 1

        last_ev = team_ev[j - 1]
        yield {
            "player": player,
            "possession": possession,
            "start_event": start_ev,
            "last_event": last_ev,
            "global_last_idx": global_index_lookup.get(last_ev.get("index")),
        }
        i = j


def pressure_per_player(match_ids, team, zip_paths):
    """
    Consolidate consecutive under-pressure events into player episodes.

    For each player:
      - count pressured episodes
      - separate successful outcomes (pass, shot, dribble escape, duel escape)
      - count primary mistakes and additional negative outcomes
      - for mistakes, track if the same player fouled / got booked before
        Olympiacos regained possession, and whether the team conceded
    """
    events_by_match = load_match_events(match_ids, zip_paths)

    stats = defaultdict(lambda: {
        "under_pressure_episodes": 0,
        "pass_success": 0,
        "shot": 0,
        "dribble_escape": 0,
        "duel_escape": 0,
        "other_success": 0,
        "primary_mistake": 0,
        "other_mistake": 0,
        "mistake_total": 0,
        "fouled_after_mistake": 0,
        "booked_after_mistake": 0,
        "conceded_after_mistake": 0,
        "matches_played": set(),
        "other_success_detail": defaultdict(int),
        "other_mistake_detail": defaultdict(int),
    })

    for mid, events in events_by_match.items():
        for episode in iter_pressure_episodes(events, team):
            player = episode["player"]
            last_ev = episode["last_event"]
            bucket, descriptor = classify_pressure_episode(last_ev)
            player_stats = stats[player]

            player_stats["under_pressure_episodes"] += 1
            player_stats["matches_played"].add(mid)

            if bucket in {"pass_success", "shot", "dribble_escape", "duel_escape"}:
                player_stats[bucket] += 1
            elif bucket == "other_success":
                player_stats["other_success"] += 1
                player_stats["other_success_detail"][descriptor] += 1
            elif bucket in {"primary_mistake", "other_mistake"}:
                player_stats[bucket] += 1
                player_stats["mistake_total"] += 1
                if bucket == "other_mistake":
                    player_stats["other_mistake_detail"][descriptor] += 1

                global_last_idx = episode["global_last_idx"]
                if global_last_idx is not None:
                    regain_idx = first_possession_regain_index(events, global_last_idx + 1, team)
                    fallout = scan_mistake_fallout(
                        events,
                        global_last_idx + 1,
                        regain_idx,
                        team,
                        player,
                    )
                    for key, happened in fallout.items():
                        if happened:
                            player_stats[key] += 1

    rows = []
    for player, s in stats.items():
        success_total = (
            s["pass_success"] +
            s["shot"] +
            s["dribble_escape"] +
            s["duel_escape"] +
            s["other_success"]
        )
        rows.append({
            "player": player,
            "under_pressure_episodes": s["under_pressure_episodes"],
            "successful_episodes": success_total,
            "success_under_pressure_pct": round(
                success_total / s["under_pressure_episodes"] * 100, 1
            ) if s["under_pressure_episodes"] > 0 else 0.0,
            "pass_success": s["pass_success"],
            "shot": s["shot"],
            "dribble_escape": s["dribble_escape"],
            "duel_escape": s["duel_escape"],
            "other_success": s["other_success"],
            "primary_mistake": s["primary_mistake"],
            "other_mistake": s["other_mistake"],
            "mistake_total": s["mistake_total"],
            "mistake_under_pressure_pct": round(
                s["mistake_total"] / s["under_pressure_episodes"] * 100, 1
            ) if s["under_pressure_episodes"] > 0 else 0.0,
            "fouled_after_mistake": s["fouled_after_mistake"],
            "booked_after_mistake": s["booked_after_mistake"],
            "conceded_after_mistake": s["conceded_after_mistake"],
            "other_success_detail": summarize_counter(s["other_success_detail"]),
            "other_mistake_detail": summarize_counter(s["other_mistake_detail"]),
            "matches_played": len(s["matches_played"]),
        })

    return pd.DataFrame(rows).sort_values(
        ["under_pressure_episodes", "mistake_total"],
        ascending=[False, False],
    )


def conceded_pressure_mistake_events(match_ids, team, zip_paths):
    """
    Return full pressured-mistake events where the subsequent opponent phase
    ended in a goal or penalty conceded before the team regained possession.
    """
    events_by_match = load_match_events(match_ids, zip_paths)
    conceded_events = []

    for mid, events in events_by_match.items():
        for episode in iter_pressure_episodes(events, team):
            last_ev = episode["last_event"]
            bucket, _descriptor = classify_pressure_episode(last_ev)
            if bucket not in {"primary_mistake", "other_mistake"}:
                continue

            global_last_idx = episode["global_last_idx"]
            if global_last_idx is None:
                continue

            regain_idx = first_possession_regain_index(events, global_last_idx + 1, team)
            concession = goal_or_penalty_conceded_before_regain(
                events,
                global_last_idx + 1,
                regain_idx,
                team,
            )
            if concession["goal_or_penalty_conceded"]:
                reasons = []
                if concession["goal_conceded"]:
                    reasons.append("goal")
                if concession["penalty_conceded"]:
                    reasons.append("penalty")
                conceded_events.append({
                    "match_id": mid,
                    "player": episode["player"],
                    "event_index": last_ev.get("index"),
                    "event_type": event_type_name(last_ev),
                    "reason": "+".join(reasons),
                    "event": last_ev,
                })

    return conceded_events


def physical_stats(match_ids, team, zip_path):
    """
    Computes per-player physical/defensive indicators:
      - miscontrol_count        : type 38 events
      - miscontrol_under_pressure : miscontrols while under_pressure == True
      - dispossessed_count      : type 3
      - foul_committed_count    : type 22
      - aerial_duel_lost        : duel type 10 (Aerial Lost)
      - aerial_duel_won         : duel outcome 4 (Won) where type 10
      - pressure_applied        : type 17 (Pressure events the player initiated)

    Returns a DataFrame indexed by player name.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    stats = defaultdict(lambda: {
        "miscontrol": 0,
        "miscontrol_under_pressure": 0,
        "dispossessed": 0,
        "foul_committed": 0,
        "aerial_lost": 0,
        "aerial_won": 0,
        "pressure_applied": 0,
        "minutes_approx": set(),
        "matches_played": set(),
    })

    for _mid, ev in team_ev:
        player = ev.get("player", {}).get("name")
        if not player:
            continue

        etype = ev.get("type", {}).get("id")
        under_p = ev.get("under_pressure", False)
        minute = ev.get("minute", 0)
        stats[player]["minutes_approx"].add((_mid, minute))
        stats[player]["matches_played"].add(_mid)

        if etype == 38:  # Miscontrol
            stats[player]["miscontrol"] += 1
            if under_p:
                stats[player]["miscontrol_under_pressure"] += 1

        elif etype == 3:  # Dispossessed
            stats[player]["dispossessed"] += 1

        elif etype == 22:  # Foul Committed
            stats[player]["foul_committed"] += 1

        elif etype == 17:  # Pressure
            stats[player]["pressure_applied"] += 1

        elif etype == 4:  # Duel
            duel = ev.get("duel", {})
            dtype = duel.get("type", {}).get("id")
            doutcome = duel.get("outcome", {}).get("id")
            if dtype == 10:  # Aerial Lost
                stats[player]["aerial_lost"] += 1
            if doutcome == 4:  # Won
                stats[player]["aerial_won"] += 1

    rows = []
    for player, s in stats.items():
        aerial_total = s["aerial_lost"] + s["aerial_won"]
        rows.append({
            "player": player,
            "miscontrol": s["miscontrol"],
            "miscontrol_under_pressure": s["miscontrol_under_pressure"],
            "pressure_rate": round(
                s["miscontrol_under_pressure"] / s["miscontrol"], 2
            ) if s["miscontrol"] > 0 else 0.0,
            "dispossessed": s["dispossessed"],
            "foul_committed": s["foul_committed"],
            "aerial_won": s["aerial_won"],
            "aerial_lost": s["aerial_lost"],
            "aerial_win_pct": round(s["aerial_won"] / aerial_total * 100, 1)
                              if aerial_total > 0 else 0.0,
            "pressure_applied": s["pressure_applied"],
            "minutes_approx": len(s["minutes_approx"]),
            "matches_played": len(s["matches_played"]),
        })

    df = pd.DataFrame(rows).sort_values("miscontrol", ascending=False)
    return df


def error_by_pitch_zone(match_ids, team, zip_path, bins_x=6, bins_y=4):
    """
    Count misctrols + dispossessions per zone on the pitch.
    Pitch is 120x80 in StatsBomb coordinates.
    Returns a (bins_x x bins_y) DataFrame with counts.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    zone_counts = np.zeros((bins_x, bins_y))

    for _mid, ev in team_ev:
        etype = ev.get("type", {}).get("id")
        if etype not in (38, 3):
            continue
        loc = ev.get("location")
        if not loc:
            continue
        x, y = loc[0], loc[1]
        xi = min(int(x / 120 * bins_x), bins_x - 1)
        yi = min(int(y / 80 * bins_y), bins_y - 1)
        zone_counts[xi][yi] += 1

    x_labels = [f"x{i*20}-{(i+1)*20}" for i in range(bins_x)]
    y_labels = [f"y{i*20}-{(i+1)*20}" for i in range(bins_y)]
    return pd.DataFrame(zone_counts.astype(int), index=x_labels, columns=y_labels)


def error_by_minute(match_ids, team, zip_path, bins=10):
    """
    Aggregate miscontrol + dispossessed events across 10-minute intervals.
    Returns a DataFrame with columns: interval, error_count.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    intervals = defaultdict(int)
    for _mid, ev in team_ev:
        etype = ev.get("type", {}).get("id")
        if etype not in (38, 3):
            continue
        minute = ev.get("minute", 0)
        bucket = (minute // bins) * bins
        label = f"{bucket}-{bucket + bins}'"
        intervals[label] += 1

    rows = sorted(intervals.items(), key=lambda x: int(x[0].split("-")[0]))
    return pd.DataFrame(rows, columns=["interval", "error_count"])



def technical_stats(match_ids, team, zip_path):
    """
    Per-player technical indicators:
      - shots, goals, xG_total, xG_per_shot
      - key_passes (shot_assist=True), goal_assists
      - pass_attempts, pass_complete, pass_accuracy
      - dribble_attempts, dribble_complete, dribble_pct
      - obv_total_net (sum across all events)
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    stats = defaultdict(lambda: {
        "shots": 0, "goals": 0, "xG": 0.0,
        "key_passes": 0, "goal_assists": 0,
        "pass_attempts": 0, "pass_complete": 0,
        "dribble_attempts": 0, "dribble_complete": 0,
        "obv_net": 0.0,
    })

    for _mid, ev in team_ev:
        player = ev.get("player", {}).get("name")
        if not player:
            continue

        etype = ev.get("type", {}).get("id")
        obv = ev.get("obv_total_net") or 0.0
        stats[player]["obv_net"] += obv

        # shots
        if etype == 16:
            shot = ev.get("shot", {})
            stats[player]["shots"] += 1
            stats[player]["xG"] += shot.get("statsbomb_xg", 0.0) or 0.0
            if shot.get("outcome", {}).get("id") == 97:  # Goal
                stats[player]["goals"] += 1

        # passes
        elif etype == 30:
            p = ev.get("pass", {})
            stats[player]["pass_attempts"] += 1
            # incomplete outcomes: 9, 74, 75, 76, 77
            outcome_id = p.get("outcome", {}).get("id")
            if outcome_id is None:  # no outcome = complete
                stats[player]["pass_complete"] += 1
            if p.get("shot_assist"):
                stats[player]["key_passes"] += 1
            if p.get("goal_assist"):
                stats[player]["goal_assists"] += 1

        # dribbles
        elif etype == 14:
            d = ev.get("dribble", {})
            stats[player]["dribble_attempts"] += 1
            if d.get("outcome", {}).get("id") == 8:  # Complete
                stats[player]["dribble_complete"] += 1

    rows = []
    for player, s in stats.items():
        rows.append({
            "player": player,
            "shots": s["shots"],
            "goals": s["goals"],
            "xG": round(s["xG"], 2),
            "xG_per_shot": round(s["xG"] / s["shots"], 3) if s["shots"] > 0 else 0.0,
            "goals_minus_xG": round(s["goals"] - s["xG"], 2),
            "key_passes": s["key_passes"],
            "goal_assists": s["goal_assists"],
            "pass_attempts": s["pass_attempts"],
            "pass_accuracy": round(s["pass_complete"] / s["pass_attempts"] * 100, 1)
                             if s["pass_attempts"] > 0 else 0.0,
            "dribble_attempts": s["dribble_attempts"],
            "dribble_success_pct": round(
                s["dribble_complete"] / s["dribble_attempts"] * 100, 1
            ) if s["dribble_attempts"] > 0 else 0.0,
            "obv_net": round(s["obv_net"], 4),
        })

    return pd.DataFrame(rows).sort_values("xG", ascending=False)


def shot_decision_quality(match_ids, team, zip_path, xg_threshold=0.15):
    """
    Identify situations when the player chose to shoot with low xG
    (shot.statsbomb_xg < xg_threshold) - potentially a pass would have been better.
    Plus viceversa: when he passed but the xG of the situation was high.

    Return 2 DataFrames: poor_shots, missed_shots (passes from zones with big xG).
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    poor_shots = []
    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 16:
            continue
        shot = ev.get("shot", {})
        xg = shot.get("statsbomb_xg", 0.0) or 0.0
        if xg < xg_threshold:
            poor_shots.append({
                "player": ev.get("player", {}).get("name"),
                "minute": ev.get("minute"),
                "xG": round(xg, 3),
                "outcome": shot.get("outcome", {}).get("name"),
                "location": ev.get("location"),
                "technique": shot.get("technique", {}).get("name"),
            })

    return pd.DataFrame(poor_shots).sort_values("xG")



def build_pass_network(match_ids, team, zip_path, min_passes=3):
    """
    Build pass network between players.
    Return a DataFrame of edges: passer → receiver, with num_passes and
    avg_obv (average on-ball value of the respective passes).

    min_passes: filter pair of players with less than N passes (reduce noise).
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    edges = defaultdict(lambda: {"count": 0, "obv_sum": 0.0})

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        passer = ev.get("player", {}).get("name")
        recipient = p.get("recipient", {}).get("name") if p.get("recipient") else None
        if not passer or not recipient:
            continue
        # skip incomplete passes
        if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77):
            continue

        key = (passer, recipient)
        edges[key]["count"] += 1
        edges[key]["obv_sum"] += ev.get("obv_total_net") or 0.0

    rows = []
    for (passer, recipient), data in edges.items():
        if data["count"] >= min_passes:
            rows.append({
                "passer": passer,
                "recipient": recipient,
                "num_passes": data["count"],
                "avg_obv": round(data["obv_sum"] / data["count"], 4),
            })

    return pd.DataFrame(rows).sort_values("num_passes", ascending=False)


def pass_cluster_profile(match_ids, team, zip_path):
    """
    Aggregate passes by cluster_id.
    Provides a 'style fingerprint' of the team: what types of passes they prefer.
    Returns a DataFrame with: cluster_id, cluster_label, count, pct.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    cluster_counts = defaultdict(lambda: {"count": 0, "label": ""})

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        cid = p.get("pass_cluster_id")
        clabel = p.get("pass_cluster_label", "")
        if cid is None:
            continue
        cluster_counts[cid]["count"] += 1
        if clabel:
            cluster_counts[cid]["label"] = clabel

    total = sum(v["count"] for v in cluster_counts.values())
    rows = [
        {
            "cluster_id": cid,
            "cluster_label": v["label"],
            "count": v["count"],
            "pct": round(v["count"] / total * 100, 1) if total > 0 else 0.0,
        }
        for cid, v in cluster_counts.items()
    ]
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def player_flow_centrality(pass_network_df):
    """
    Calculate simplified 'flow centrality' for each player:
    how many passes they initiated + received (weighted by num_passes).
    Proxy for importance in the passing network.
    """
    out_flow = pass_network_df.groupby("passer")["num_passes"].sum().rename("passes_out")
    in_flow = pass_network_df.groupby("recipient")["num_passes"].sum().rename("passes_in")
    df = pd.concat([out_flow, in_flow], axis=1).fillna(0)
    df["flow_centrality"] = df["passes_out"] + df["passes_in"]
    df["pass_out_pct"] = round(df["passes_out"] / df["flow_centrality"] * 100, 1)
    return df.sort_values("flow_centrality", ascending=False).reset_index().rename(
        columns={"index": "player"}
    )


GOAL_CENTER = (120, 40)  # StatsBomb pitch coordinates

def distance_to_goal(x, y):
    return round(np.sqrt((x - GOAL_CENTER[0])**2 + (y - GOAL_CENTER[1])**2), 2)


def fatigue_proxy(match_ids, team, zip_path):
    """
    Compares error rate (miscontrol + dispossessed) in minutes 0-30 vs 70-90
    per player. A higher late_rate suggests fatigue-related drop in performance.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    stats = defaultdict(lambda: {"early": 0, "late": 0, "early_events": 0, "late_events": 0})

    for _mid, ev in team_ev:
        player = ev.get("player", {}).get("name")
        if not player:
            continue

        minute = ev.get("minute", 0)
        etype  = ev.get("type", {}).get("id")
        is_error = etype in (38, 3)  # miscontrol or dispossessed

        if minute <= 30:
            stats[player]["early_events"] += 1
            if is_error:
                stats[player]["early"] += 1
        elif minute >= 70:
            stats[player]["late_events"] += 1
            if is_error:
                stats[player]["late"] += 1

    rows = []
    for player, s in stats.items():
        early_rate = round(s["early"] / s["early_events"] * 100, 2) if s["early_events"] > 0 else 0.0
        late_rate  = round(s["late"]  / s["late_events"]  * 100, 2) if s["late_events"]  > 0 else 0.0
        rows.append({
            "player": player,
            "errors_0_30": s["early"],
            "events_0_30": s["early_events"],
            "error_rate_0_30": early_rate,
            "errors_70_90": s["late"],
            "events_70_90": s["late_events"],
            "error_rate_70_90": late_rate,
            "fatigue_delta": round(late_rate - early_rate, 2),  # positive = worse late
        })

    return pd.DataFrame(rows).sort_values("fatigue_delta", ascending=False)



def injury_proneness(match_ids, team, zip_path):
    """
    Counts permanent Player Off events (type 27, Permanent=True) per player.
    Also counts total substitutions for context.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    stats = defaultdict(lambda: {"injuries": 0, "substitutions": 0})

    for _mid, ev in team_ev:
        player = ev.get("player", {}).get("name")
        if not player:
            continue

        etype = ev.get("type", {}).get("id")

        if etype == 27:  # Player Off
            player_off = ev.get("player_off", {})
            if ev.get("permanent") or player_off.get("permanent"):
                stats[player]["injuries"] += 1

        elif etype == 19:  # Substitution
            stats[player]["substitutions"] += 1

    rows = [
        {
            "player": player,
            "injuries": s["injuries"],
            "substitutions": s["substitutions"],
        }
        for player, s in stats.items()
        if s["injuries"] > 0 or s["substitutions"] > 0
    ]

    return pd.DataFrame(rows).sort_values("injuries", ascending=False)



def shot_distance_stats(match_ids, team, zip_path):
    """
    Per player: average distance to goal at time of shot and at time of goal.
    Also includes xG for context.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    stats = defaultdict(lambda: {
        "shot_distances": [], "goal_distances": [], "xg_values": []
    })

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 16:
            continue
        player = ev.get("player", {}).get("name")
        if not player:
            continue
        loc = ev.get("location")
        if not loc:
            continue

        dist = distance_to_goal(loc[0], loc[1])
        xg   = ev.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
        stats[player]["shot_distances"].append(dist)
        stats[player]["xg_values"].append(xg)

        if ev.get("shot", {}).get("outcome", {}).get("id") == 97:  # Goal
            stats[player]["goal_distances"].append(dist)

    rows = []
    for player, s in stats.items():
        if not s["shot_distances"]:
            continue
        rows.append({
            "player": player,
            "shots": len(s["shot_distances"]),
            "goals": len(s["goal_distances"]),
            "avg_shot_distance": round(np.mean(s["shot_distances"]), 2),
            "avg_goal_distance": round(np.mean(s["goal_distances"]), 2) if s["goal_distances"] else None,
            "min_shot_distance": round(np.min(s["shot_distances"]), 2),
            "max_shot_distance": round(np.max(s["shot_distances"]), 2),
            "avg_xg": round(np.mean(s["xg_values"]), 3),
        })

    return pd.DataFrame(rows).sort_values("shots", ascending=False)



def xg_conditioned_on_passer(match_ids, team, zip_path):
    """
    E[goal | pass from player X]:
    For each player who assisted a shot, compute the average xG of shots
    that followed their pass, and how many resulted in goals.
    Uses assisted_shot_id to link pass -> shot.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    # build shot lookup: shot_id -> (xg, is_goal)
    shot_lookup = {}
    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 16:
            continue
        shot_id = ev.get("id")
        xg = ev.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
        is_goal = ev.get("shot", {}).get("outcome", {}).get("id") == 97
        shot_lookup[shot_id] = {"xg": xg, "is_goal": is_goal}

    stats = defaultdict(lambda: {"xg_values": [], "goals": 0, "assists": 0})

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        assisted_shot_id = p.get("assisted_shot_id")
        if not assisted_shot_id:
            continue
        passer = ev.get("player", {}).get("name")
        if not passer:
            continue
        shot = shot_lookup.get(assisted_shot_id)
        if not shot:
            continue

        stats[passer]["xg_values"].append(shot["xg"])
        stats[passer]["assists"] += 1
        if shot["is_goal"]:
            stats[passer]["goals"] += 1

    rows = []
    for player, s in stats.items():
        if not s["xg_values"]:
            continue
        rows.append({
            "player": player,
            "shot_assists": s["assists"],
            "goals_assisted": s["goals"],
            "avg_xg_created": round(np.mean(s["xg_values"]), 3),
            "total_xg_created": round(sum(s["xg_values"]), 3),
            "conversion_rate": round(s["goals"] / s["assists"] * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("total_xg_created", ascending=False)



def corner_analysis(match_ids, team, zip_path):
    """
    Per player who took corners:
    - corners taken
    - corners leading to a shot (via assisted_shot_id or shot_assist flag)
    - corners leading to a goal
    - conversion rate

    Also returns a summary row for the whole team.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    shot_lookup = {}
    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 16:
            continue
        shot_id = ev.get("id")
        is_goal = ev.get("shot", {}).get("outcome", {}).get("id") == 97
        shot_lookup[shot_id] = is_goal

    stats = defaultdict(lambda: {"taken": 0, "led_to_shot": 0, "led_to_goal": 0})

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        if p.get("type", {}).get("id") != 61:
            continue

        player = ev.get("player", {}).get("name")
        if not player:
            continue

        stats[player]["taken"] += 1

        assisted_shot_id = p.get("assisted_shot_id")
        shot_assist = p.get("shot_assist", False)
        goal_assist = p.get("goal_assist", False)

        if assisted_shot_id or shot_assist:
            stats[player]["led_to_shot"] += 1
            if goal_assist or (assisted_shot_id and shot_lookup.get(assisted_shot_id)):
                stats[player]["led_to_goal"] += 1

    rows = []
    for player, s in stats.items():
        rows.append({
            "player": player,
            "corners_taken": s["taken"],
            "led_to_shot": s["led_to_shot"],
            "led_to_goal": s["led_to_goal"],
            "shot_rate_pct": round(s["led_to_shot"] / s["taken"] * 100, 1) if s["taken"] > 0 else 0.0,
            "goal_rate_pct": round(s["led_to_goal"] / s["taken"] * 100, 1) if s["taken"] > 0 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("corners_taken", ascending=False)

    summary = pd.DataFrame([{
        "player": "TEAM TOTAL",
        "corners_taken": df["corners_taken"].sum(),
        "led_to_shot": df["led_to_shot"].sum(),
        "led_to_goal": df["led_to_goal"].sum(),
        "shot_rate_pct": round(df["led_to_shot"].sum() / df["corners_taken"].sum() * 100, 1)
                         if df["corners_taken"].sum() > 0 else 0.0,
        "goal_rate_pct": round(df["led_to_goal"].sum() / df["corners_taken"].sum() * 100, 1)
                         if df["corners_taken"].sum() > 0 else 0.0,
    }])

    return pd.concat([df, summary], ignore_index=True)


def csv_to_markdown(csv_path, output_path=None):
    df = pd.read_csv(csv_path)
    md = df.to_markdown(index=False)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(md)
        print(f"Saved: {output_path}")
    else:
        print(md)


if __name__ == "__main__":
    matches_df = pd.read_csv(MATCHES_CSV)
    team_matches = matches_df[
        (matches_df["home"] == TEAM_MATCHES_CSV) | (matches_df["away"] == TEAM_MATCHES_CSV)
    ]
    all_ids = team_matches["statsbomb"].tolist()
    print(f"{TEAM_MATCHES_CSV}: {len(all_ids)} games")

    available = set()
    for one_zip_path in ZIP_PATHS:
        with zipfile.ZipFile(one_zip_path) as zf:
            available.update(zf.namelist())

    match_ids = [mid for mid in all_ids if f"{mid}.json" in available]
    missing = [mid for mid in all_ids if f"{mid}.json" not in available]

    print(f"{TEAM_MATCHES_CSV}: {len(match_ids)}/{len(all_ids)} available matches")
    if missing:
        print(f"Missing from zip: {missing}")

    # 1. Physical
    print("=== MODULE 1: Physical Stats ===")
    phys = physical_stats(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(phys.to_markdown(index=False))

    print("\n--- Errors by pitch zone ---")
    zone_df = error_by_pitch_zone(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(zone_df.to_markdown())

    print("\n--- Errors by minute ---")
    time_df = error_by_minute(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(time_df.to_markdown(index=False))

    # 2. Technical
    print("\n=== MODULE 2: Technical Stats ===")
    tech = technical_stats(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(tech.to_markdown(index=False))

    print("\n--- Shots with low xG (< 0.10) ---")
    poor = shot_decision_quality(match_ids, TEAM_STATSBOMB, ZIP_PATHS, xg_threshold=0.10)
    print(poor.to_markdown(index=False))

    # 3. Passing network
    print("\n=== MODULE 3: Passing Network ===")
    net = build_pass_network(match_ids, TEAM_STATSBOMB, ZIP_PATHS, min_passes=5)
    print(net.head(20).to_markdown(index=False))

    centrality = player_flow_centrality(net)
    print("\n--- Flow Centrality ---")
    print(centrality.to_markdown(index=False))

    print("\n--- Pass Cluster Profile ---")
    clusters = pass_cluster_profile(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(clusters.head(15).to_markdown(index=False))


    # save everything to csv
    OUTPUT_DIR = "output/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    phys.to_csv(os.path.join(OUTPUT_DIR, "physical_stats.csv"),    index=False)
    zone_df.to_csv(os.path.join(OUTPUT_DIR, "errors_by_zone.csv"), index=False)
    time_df.to_csv(os.path.join(OUTPUT_DIR, "errors_by_minute.csv"), index=False)
    tech.to_csv(os.path.join(OUTPUT_DIR, "technical_stats.csv"),   index=False)
    poor.to_csv(os.path.join(OUTPUT_DIR, "poor_shots.csv"),        index=False)
    net.to_csv(os.path.join(OUTPUT_DIR, "pass_network.csv"),       index=False)
    centrality.to_csv(os.path.join(OUTPUT_DIR, "flow_centrality.csv"), index=False)
    clusters.to_csv(os.path.join(OUTPUT_DIR, "pass_clusters.csv"), index=False)


    # physical proxy
    print("\n=== FATIGUE PROXY ===")
    fatigue = fatigue_proxy(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(fatigue.to_markdown(index=False))

    print("\n=== INJURY PRONENESS ===")
    injuries = injury_proneness(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(injuries.to_markdown(index=False))

    # technical extras
    print("\n=== SHOT DISTANCE STATS ===")
    shot_dist = shot_distance_stats(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(shot_dist.to_markdown(index=False))

    print("\n=== XG CONDITIONED ON PASSER ===")
    xg_passer = xg_conditioned_on_passer(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(xg_passer.to_markdown(index=False))

    print("\n=== CORNER ANALYSIS ===")
    corners = corner_analysis(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(corners.to_markdown(index=False))

    print("\n=== PRESSURE PER PLAYER ===")
    pressure_df = pressure_per_player(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    print(pressure_df.to_markdown(index=False))

    print("\n=== PRESSURED MISTAKES THAT LED TO GOALS OR PENALTIES CONCEDED ===")
    conceded_events = conceded_pressure_mistake_events(match_ids, TEAM_STATSBOMB, ZIP_PATHS)
    if conceded_events:
        for item in conceded_events:
            print(
                f"match {item['match_id']} | player {item['player']} | "
                f"event_index {item['event_index']} | type {item['event_type']} | "
                f"reason {item['reason']}"
            )
            print(json.dumps(item["event"], ensure_ascii=True, indent=2))
    else:
        print("No pressured mistakes leading to conceded goals or penalties found.")

    # save to CSV
    fatigue.to_csv(os.path.join(OUTPUT_DIR, "fatigue_proxy.csv"), index=False)
    injuries.to_csv(os.path.join(OUTPUT_DIR, "injury_proneness.csv"), index=False)
    shot_dist.to_csv(os.path.join(OUTPUT_DIR, "shot_distance.csv"), index=False)
    xg_passer.to_csv(os.path.join(OUTPUT_DIR, "xg_by_passer.csv"), index=False)
    corners.to_csv(os.path.join(OUTPUT_DIR, "corner_analysis.csv"), index=False)
    pressure_df.to_csv(os.path.join(OUTPUT_DIR, "pressure_per_payer.csv"), index=False)

    print(f"Saved CSVs to {OUTPUT_DIR}/")


    # save as markdown as well
    csv_to_markdown(os.path.join(OUTPUT_DIR, "physical_stats.csv"),
                    os.path.join(OUTPUT_DIR, "physical_stats.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "technical_stats.csv"),
                    os.path.join(OUTPUT_DIR, "technical_stats.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "poor_shots.csv"),
                    os.path.join(OUTPUT_DIR, "poor_shots.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "pass_network.csv"),
                    os.path.join(OUTPUT_DIR, "pass_network.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "flow_centrality.csv"),
                    os.path.join(OUTPUT_DIR, "flow_centrality.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "pass_clusters.csv"),
                    os.path.join(OUTPUT_DIR, "pass_clusters.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "fatigue_proxy.csv"),
                    os.path.join(OUTPUT_DIR, "fatigue_proxy.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "injury_proneness.csv"),
                    os.path.join(OUTPUT_DIR, "injury_proneness.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "shot_distance.csv"),
                    os.path.join(OUTPUT_DIR, "shot_distance.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "xg_by_passer.csv"),
                    os.path.join(OUTPUT_DIR, "xg_by_passer.md"))
    csv_to_markdown(os.path.join(OUTPUT_DIR, "corner_analysis.csv"),
                    os.path.join(OUTPUT_DIR, "corner_analysis.md"))



'''
Accumulated data:

Physical Stats:

miscontrol count per player
miscontrol under pressure count per player
pressure rate (miscontrol under pressure / total miscontrol)
dispossessed count per player
foul committed count per player
aerial duels won/lost per player
aerial win percentage per player
pressure applied per player
minutes played (proxy) per player
matches played per player
errors per pitch zone (heatmap grid 6x4)
errors per time interval (10-minute bins)
fatigue delta (error rate 70-90 vs 0-30 per player)
injury proneness (permanent Player Off events per player)

Technical Stats:

shots per player
goals per player
xG total per player
xG per shot per player
goals minus xG (over/underperformance) per player
key passes per player
goal assists per player
pass attempts per player
pass accuracy per player
dribble attempts per player
dribble success rate per player
OBV net per player
matches played per player
shot distance stats (avg/min/max distance to goal at shot, avg distance at goal)
xG conditioned on passer (avg xG created, total xG created, conversion rate)
poor shots (shots with xG < 0.10)
corner analysis (corners taken, led to shot, led to goal, rates)

Passing Network:

pass network edges (passer → recipient, volume, avg OBV)
flow centrality per player
pass cluster profile (style fingerprint per cluster)

'''
