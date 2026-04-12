from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from utils import (
    POSITION_ABBREVIATIONS,
    TEAM_STATSBOMB,
    ZIP_PATHS,
    competition_match_ids,
    competition_player_profiles,
    draw_pitch,
    load_events,
    load_match_events,
    make_heatmap,
    most_common_value,
)

CMAP_BLUE = LinearSegmentedColormap.from_list(
    "blue_hm",
    ["#1a1a2e", "#0d2b5e", "#1565c0", "#42a5f5", "#b3e5fc"],
)


def event_type_name(ev):
    return ev.get("type", {}).get("name") or "Unknown"


def player_name(ev):
    return ev.get("player", {}).get("name")


def event_descriptor(ev):
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

    return "other_success", descriptor


def first_possession_regain_index(events, start_idx, team):
    for idx in range(start_idx, len(events)):
        if events[idx].get("possession_team", {}).get("name") == team:
            return idx
    return len(events)


def scan_mistake_fallout(events, start_idx, regain_idx, team, player):
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
    team_ev = [ev for ev in events if ev.get("team", {}).get("name") == team and player_name(ev)]
    global_index_lookup = {ev.get("index"): idx for idx, ev in enumerate(events)}

    i = 0
    while i < len(team_ev):
        start_ev = team_ev[i]
        if not start_ev.get("under_pressure"):
            i += 1
            continue

        player = player_name(start_ev)
        player_id = start_ev.get("player", {}).get("id")
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
            "player_id": player_id,
            "player": player,
            "possession": possession,
            "start_event": start_ev,
            "last_event": last_ev,
            "global_last_idx": global_index_lookup.get(last_ev.get("index")),
        }
        i = j


def empty_pressure_statline():
    return {
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
        "team_counts": Counter(),
        "other_success_detail": defaultdict(int),
        "other_mistake_detail": defaultdict(int),
    }


def apply_pressure_episode(stats, episode, events, team, match_id):
    key = (episode["player_id"], episode["player"])
    player_stats = stats[key]
    last_ev = episode["last_event"]
    bucket, descriptor = classify_pressure_episode(last_ev)

    player_stats["under_pressure_episodes"] += 1
    player_stats["matches_played"].add(match_id)
    player_stats["team_counts"][team] += 1

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
                episode["player"],
            )
            for field, happened in fallout.items():
                if happened:
                    player_stats[field] += 1


def pressure_dataframe_from_stats(stats, include_team=False):
    rows = []
    for (player_id, player), statline in stats.items():
        success_total = (
            statline["pass_success"] +
            statline["shot"] +
            statline["dribble_escape"] +
            statline["duel_escape"] +
            statline["other_success"]
        )
        row = {
            "player_id": player_id,
            "player": player,
            "under_pressure_episodes": statline["under_pressure_episodes"],
            "successful_episodes": success_total,
            "success_under_pressure_pct": round(
                success_total / statline["under_pressure_episodes"] * 100, 1
            ) if statline["under_pressure_episodes"] > 0 else 0.0,
            "pass_success": statline["pass_success"],
            "shot": statline["shot"],
            "dribble_escape": statline["dribble_escape"],
            "duel_escape": statline["duel_escape"],
            "other_success": statline["other_success"],
            "primary_mistake": statline["primary_mistake"],
            "other_mistake": statline["other_mistake"],
            "mistake_total": statline["mistake_total"],
            "mistake_under_pressure_pct": round(
                statline["mistake_total"] / statline["under_pressure_episodes"] * 100, 1
            ) if statline["under_pressure_episodes"] > 0 else 0.0,
            "fouled_after_mistake": statline["fouled_after_mistake"],
            "booked_after_mistake": statline["booked_after_mistake"],
            "conceded_after_mistake": statline["conceded_after_mistake"],
            "other_success_detail": summarize_counter(statline["other_success_detail"]),
            "other_mistake_detail": summarize_counter(statline["other_mistake_detail"]),
            "matches_played": len(statline["matches_played"]),
        }
        if include_team:
            row["team"] = most_common_value(statline["team_counts"])
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["under_pressure_episodes", "mistake_total", "player"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def pressure_per_player(match_ids, team, zip_paths):
    events_by_match = load_match_events(match_ids, zip_paths)
    stats = defaultdict(empty_pressure_statline)

    for mid, events in events_by_match.items():
        for episode in iter_pressure_episodes(events, team):
            apply_pressure_episode(stats, episode, events, team, mid)

    return pressure_dataframe_from_stats(stats)


def competition_pressure_per_player(match_ids, zip_paths):
    events_by_match = load_match_events(match_ids, zip_paths)
    stats = defaultdict(empty_pressure_statline)

    for mid, events in events_by_match.items():
        teams_in_match = sorted(
            {
                ev.get("team", {}).get("name")
                for ev in events
                if ev.get("team", {}).get("name")
            }
        )
        for team in teams_in_match:
            for episode in iter_pressure_episodes(events, team):
                apply_pressure_episode(stats, episode, events, team, mid)

    pressure_df = pressure_dataframe_from_stats(stats, include_team=True)
    if pressure_df.empty:
        return pressure_df

    profiles_df = competition_player_profiles(match_ids, zip_paths)
    if not profiles_df.empty:
        pressure_df = pressure_df.merge(
            profiles_df,
            on=["player_id", "player"],
            how="left",
        )
        pressure_df["team"] = pressure_df["team"].fillna(pressure_df["team_profile"])
        pressure_df = pressure_df.drop(columns=["team_profile"])

    return pressure_df.sort_values(
        ["under_pressure_episodes", "mistake_total", "player"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def olympiacos_position_mistake_ranks(competition_pressure_df, olympiacos_team=TEAM_STATSBOMB):
    if competition_pressure_df.empty:
        return pd.DataFrame()

    ranked = competition_pressure_df[
        competition_pressure_df["primary_position_id"].notna()
    ].copy()
    if ranked.empty:
        return ranked

    ranked["players_compared_at_position"] = ranked.groupby("primary_position_id")["player_id"].transform("count")
    ranked["mistake_pct_rank_in_position"] = (
        ranked.groupby("primary_position_id")["mistake_under_pressure_pct"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    ranked["primary_position_id"] = ranked["primary_position_id"].astype(int)
    ranked["primary_starting_position_id"] = ranked["primary_starting_position_id"].fillna(
        ranked["primary_position_id"]
    ).astype(int)
    ranked["primary_position_abbreviation"] = ranked["primary_position_abbreviation"].fillna(
        ranked["primary_starting_position_abbreviation"]
    )

    olympiacos_rows = ranked[ranked["team"] == olympiacos_team].copy()
    olympiacos_rows = olympiacos_rows[
        [
            "player",
            "team",
            "primary_position_id",
            "primary_position_abbreviation",
            "under_pressure_episodes",
            "success_under_pressure_pct",
            "mistake_total",
            "mistake_under_pressure_pct",
            "mistake_pct_rank_in_position",
            "players_compared_at_position",
            "matches_played",
            "starts",
        ]
    ]

    return olympiacos_rows.sort_values(
        ["primary_position_id", "mistake_pct_rank_in_position", "mistake_under_pressure_pct", "player"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def competition_top10_under_pressure(competition_pressure_df, top_n=10, min_episodes=50):
    if competition_pressure_df.empty:
        return pd.DataFrame()

    filtered_df = competition_pressure_df[
        competition_pressure_df["under_pressure_episodes"] >= min_episodes
    ].copy()
    if filtered_df.empty:
        return pd.DataFrame()

    columns = [
        "player",
        "team",
        "primary_position_abbreviation",
        "under_pressure_episodes",
        "success_under_pressure_pct",
        "mistake_under_pressure_pct",
    ]
    available_columns = [column for column in columns if column in filtered_df.columns]

    return filtered_df.sort_values(
        ["success_under_pressure_pct", "under_pressure_episodes", "player"],
        ascending=[False, False, True],
    ).head(top_n)[available_columns].reset_index(drop=True)


def conceded_pressure_mistake_events(match_ids, team, zip_paths):
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


def conceded_pressure_mistake_locations(match_ids, team, zip_path):
    coords = []
    for item in conceded_pressure_mistake_events(match_ids, team, zip_path):
        loc = item["event"].get("location")
        if loc:
            coords.append((loc[0], loc[1]))
    return coords


def plot_pressure_heatmap(match_ids, team, zip_path, save_path="heatmap_pressure.png"):
    records = load_events(match_ids, zip_path)

    coords = []
    for _mid, ev in records:
        if ev.get("type", {}).get("id") != 17:
            continue
        if ev.get("team", {}).get("name") == team:
            continue
        loc = ev.get("location")
        if loc:
            coords.append((loc[0], loc[1]))

    coords = [(120 - x, 80 - y) for x, y in coords]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a1a2e")
    draw_pitch(ax)

    hm = make_heatmap(coords, sigma=4)
    ax.imshow(
        hm, extent=[0, 120, 0, 80], origin="lower",
        cmap=CMAP_BLUE, alpha=0.75, aspect="auto"
    )

    ax.set_title(
        f"{team} - zones of received pressure\n"
        f"n={len(coords)} · {len(match_ids)} matches",
        color="white", fontsize=13, pad=12, fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()


def plot_pressure_success_scatter(zip_paths, save_path="output/pressure_success_scatter.png", min_episodes=50):
    match_ids = competition_match_ids(zip_paths)
    competition_df = competition_pressure_per_player(match_ids, zip_paths)
    filtered_df = competition_df[
        competition_df["under_pressure_episodes"] >= min_episodes
    ].copy()

    if filtered_df.empty:
        print(f"No players found with at least {min_episodes} under-pressure episodes.")
        return

    filtered_df["is_olympiacos"] = filtered_df["team"] == TEAM_STATSBOMB
    top10_player_ids = set(
        filtered_df.sort_values(
            ["success_under_pressure_pct", "under_pressure_episodes", "player"],
            ascending=[False, False, True],
        ).head(10)["player_id"]
    )
    top10_volume_player_ids = set(
        filtered_df.sort_values(
            ["under_pressure_episodes", "success_under_pressure_pct", "player"],
            ascending=[False, False, True],
        ).head(10)["player_id"]
    )
    filtered_df["is_top10"] = filtered_df["player_id"].isin(top10_player_ids)
    filtered_df["is_top10_volume"] = filtered_df["player_id"].isin(top10_volume_player_ids)

    fig, ax = plt.subplots(figsize=(13, 9), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    others = filtered_df[~filtered_df["is_olympiacos"] & ~filtered_df["is_top10"]]
    top10_players = filtered_df[filtered_df["is_top10"]]
    top10_volume_players = filtered_df[filtered_df["is_top10_volume"]]
    olympiacos = filtered_df[filtered_df["is_olympiacos"]]

    if not others.empty:
        ax.scatter(
            others["under_pressure_episodes"],
            others["success_under_pressure_pct"],
            s=55,
            color="#8ecae6",
            alpha=0.55,
            edgecolors="none",
            label=f"Other players ({len(others)})",
        )

    if not top10_volume_players.empty:
        ax.scatter(
            top10_volume_players["under_pressure_episodes"],
            top10_volume_players["success_under_pressure_pct"],
            s=220,
            facecolors="none",
            edgecolors="#4cc9f0",
            linewidths=1.8,
            marker="D",
            label="Top 10 by episodes",
            zorder=4,
        )

        for row in top10_volume_players.itertuples(index=False):
            ax.annotate(
                row.player.split()[-1],
                (row.under_pressure_episodes, row.success_under_pressure_pct),
                xytext=(8, 8),
                textcoords="offset points",
                color="#4cc9f0",
                fontsize=8.5,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#1a1a2e")],
            )

    if not top10_players.empty:
        ax.scatter(
            top10_players["under_pressure_episodes"],
            top10_players["success_under_pressure_pct"],
            s=220,
            color="#90be6d",
            alpha=0.95,
            edgecolors="#f1fa8c",
            linewidths=1.2,
            marker="*",
            label="Top 10 overall",
            zorder=5,
        )

        for row in top10_players.itertuples(index=False):
            ax.annotate(
                row.player.split()[-1],
                (row.under_pressure_episodes, row.success_under_pressure_pct),
                xytext=(7, -10),
                textcoords="offset points",
                color="#f1fa8c",
                fontsize=8.5,
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#1a1a2e")],
            )

    if not olympiacos.empty:
        ax.scatter(
            olympiacos["under_pressure_episodes"],
            olympiacos["success_under_pressure_pct"],
            s=120,
            color="#ffb703",
            alpha=0.95,
            edgecolors="#d00000",
            linewidths=1.0,
            label=f"Olympiacos ({len(olympiacos)})",
            zorder=6,
        )

        for row in olympiacos.itertuples(index=False):
            ax.annotate(
                row.player.split()[-1],
                (row.under_pressure_episodes, row.success_under_pressure_pct),
                xytext=(6, 6),
                textcoords="offset points",
                color="white",
                fontsize=8,
                zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#1a1a2e")],
            )

    x_max = filtered_df["under_pressure_episodes"].max()
    y_min = max(0, filtered_df["success_under_pressure_pct"].min() - 3)
    y_max = min(100, filtered_df["success_under_pressure_pct"].max() + 3)

    ax.axvline(filtered_df["under_pressure_episodes"].median(), color="white", alpha=0.12, linestyle="--")
    ax.axhline(filtered_df["success_under_pressure_pct"].median(), color="white", alpha=0.12, linestyle="--")

    ax.set_xlim(0, x_max + 20)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("under-pressure episodes", color="white", fontsize=11)
    ax.set_ylabel("success under pressure (%)", color="white", fontsize=11)
    ax.set_title(
        "Competition Under-Pressure Profile\n"
        f"x = volume, y = success rate, min {min_episodes} episodes, upper-right is better",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")
    ax.grid(color="white", alpha=0.08, linestyle="--")
    ax.legend(facecolor="#0d0d1a", labelcolor="white", fontsize=10, framealpha=0.85, loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()
