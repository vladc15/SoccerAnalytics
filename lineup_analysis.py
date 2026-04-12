import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    MATCHES_CSV,
    TEAM_MATCHES_CSV,
    TEAM_STATSBOMB,
    csv_to_markdown,
    draw_pitch,
    team_lineup_entry,
)

OUTPUT_DIR = "output"


POSITION_INFO = {
    1: {"abbr": "GK", "name": "Goalkeeper", "xy": (8, 40)},
    2: {"abbr": "RB", "name": "Right Back", "xy": (28, 66)},
    3: {"abbr": "RCB", "name": "Right Center Back", "xy": (26, 54)},
    4: {"abbr": "CB", "name": "Center Back", "xy": (24, 40)},
    5: {"abbr": "LCB", "name": "Left Center Back", "xy": (26, 26)},
    6: {"abbr": "LB", "name": "Left Back", "xy": (28, 14)},
    7: {"abbr": "RWB", "name": "Right Wing Back", "xy": (40, 70)},
    8: {"abbr": "LWB", "name": "Left Wing Back", "xy": (40, 10)},
    9: {"abbr": "RDM", "name": "Right Defensive Midfield", "xy": (48, 54)},
    10: {"abbr": "CDM", "name": "Center Defensive Midfield", "xy": (50, 40)},
    11: {"abbr": "LDM", "name": "Left Defensive Midfield", "xy": (48, 26)},
    12: {"abbr": "RM", "name": "Right Midfield", "xy": (64, 66)},
    13: {"abbr": "RCM", "name": "Right Center Midfield", "xy": (64, 54)},
    14: {"abbr": "CM", "name": "Center Midfield", "xy": (64, 40)},
    15: {"abbr": "LCM", "name": "Left Center Midfield", "xy": (64, 26)},
    16: {"abbr": "LM", "name": "Left Midfield", "xy": (64, 14)},
    17: {"abbr": "RW", "name": "Right Wing", "xy": (80, 66)},
    18: {"abbr": "RAM", "name": "Right Attacking Midfield", "xy": (80, 54)},
    19: {"abbr": "CAM", "name": "Center Attacking Midfield", "xy": (82, 40)},
    20: {"abbr": "LAM", "name": "Left Attacking Midfield", "xy": (80, 26)},
    21: {"abbr": "LW", "name": "Left Wing", "xy": (80, 14)},
    22: {"abbr": "RCF", "name": "Right Center Forward", "xy": (104, 52)},
    23: {"abbr": "ST", "name": "Striker", "xy": (106, 40)},
    24: {"abbr": "LCF", "name": "Left Center Forward", "xy": (104, 28)},
    25: {"abbr": "SS", "name": "Secondary Striker", "xy": (92, 40)},
}


def load_olympiacos_matches():
    matches_df = pd.read_csv(MATCHES_CSV)
    olympiacos_matches = matches_df[
        (matches_df["home"] == TEAM_MATCHES_CSV) | (matches_df["away"] == TEAM_MATCHES_CSV)
    ].copy()
    olympiacos_matches["statsbomb"] = olympiacos_matches["statsbomb"].astype(str)
    olympiacos_matches["venue"] = olympiacos_matches["home"].eq(TEAM_MATCHES_CSV).map(
        {True: "home", False: "away"}
    )
    olympiacos_matches["opponent"] = olympiacos_matches.apply(
        lambda row: row["away"] if row["home"] == TEAM_MATCHES_CSV else row["home"],
        axis=1,
    )
    return olympiacos_matches
def display_name(player_entry):
    nickname = player_entry.get("player_nickname")
    if isinstance(nickname, str) and nickname.strip():
        return nickname.strip()
    return player_entry.get("player_name", "Unknown")


def format_formation(formation):
    if pd.isna(formation):
        return "Unknown"
    formation_str = str(int(formation))
    return "-".join(formation_str)


def build_starter_position_summary(match_rows):
    position_player_counts = defaultdict(Counter)
    player_meta = {}
    formation_counts = Counter()

    for row in match_rows.itertuples(index=False):
        team = team_lineup_entry(row.statsbomb, TEAM_STATSBOMB)
        starting_formation = next(
            (item.get("formation") for item in team.get("formations", []) if item.get("reason") == "Starting XI"),
            None,
        )
        formation_counts[starting_formation] += 1

        for player in team.get("lineup", []):
            positions = player.get("positions", [])
            if not positions:
                continue
            first_position = positions[0]
            if first_position.get("start_reason") != "Starting XI":
                continue

            player_name = player["player_name"]
            player_meta[player_name] = {
                "player_name": player_name,
                "display_name": display_name(player),
            }
            position_player_counts[first_position["position_id"]][player_name] += 1

    rows = []
    total_starts_by_player = Counter()
    for position_id, counts in position_player_counts.items():
        for player_name, starts_at_position in counts.items():
            total_starts_by_player[player_name] += starts_at_position

    for position_id, counts in position_player_counts.items():
        info = POSITION_INFO[position_id]
        for player_name, starts_at_position in counts.items():
            rows.append(
                {
                    "position_id": position_id,
                    "position_abbreviation": info["abbr"],
                    "position_name": info["name"],
                    "player_name": player_name,
                    "display_name": player_meta[player_name]["display_name"],
                    "starts_at_position": starts_at_position,
                    "total_starts": total_starts_by_player[player_name],
                }
            )

    starter_df = pd.DataFrame(rows)
    if not starter_df.empty:
        starter_df = starter_df.sort_values(
            ["position_id", "starts_at_position", "total_starts", "display_name"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

    formation_df = pd.DataFrame(
        [
            {
                "formation": formation,
                "formation_label": format_formation(formation),
                "matches": matches,
            }
            for formation, matches in sorted(formation_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    )

    total_starts_df = (
        starter_df.groupby(["player_name", "display_name"], as_index=False)["starts_at_position"]
        .sum()
        .rename(columns={"starts_at_position": "total_starts"})
        .sort_values(["total_starts", "display_name"], ascending=[False, True])
        .reset_index(drop=True)
        if not starter_df.empty
        else pd.DataFrame(columns=["player_name", "display_name", "total_starts"])
    )

    return starter_df, total_starts_df, formation_df


def find_subbed_off_position(lineup_players, incoming_player_name, outgoing_player_name):
    for player in lineup_players:
        if player.get("player_name") != outgoing_player_name:
            continue
        for position in player.get("positions", []):
            if not str(position.get("end_reason", "")).startswith("Substitution - Off"):
                continue
            if position.get("counterpart_name") == incoming_player_name:
                return position
    return None


def build_substitutions_table(match_rows):
    rows = []

    for row in match_rows.itertuples(index=False):
        team = team_lineup_entry(row.statsbomb, TEAM_STATSBOMB)
        lineup_players = team.get("lineup", [])

        for player in lineup_players:
            for position in player.get("positions", []):
                if not str(position.get("start_reason", "")).startswith("Substitution - On"):
                    continue

                outgoing_player = position.get("counterpart_name")
                outgoing_position = find_subbed_off_position(
                    lineup_players,
                    incoming_player_name=player.get("player_name"),
                    outgoing_player_name=outgoing_player,
                )

                rows.append(
                    {
                        "opponent": row.opponent,
                        "player_in": player.get("player_name"),
                        "player_in_position": POSITION_INFO.get(
                            position.get("position_id"), {}
                        ).get("abbr"),
                        "player_off": outgoing_player,
                        "player_off_position": POSITION_INFO.get(
                            outgoing_position.get("position_id"), {}
                        ).get("abbr")
                        if outgoing_position
                        else None,
                        "substitution_reason": normalize_substitution_reason(position.get("start_reason")),
                    }
                )

    substitutions_df = pd.DataFrame(rows)
    if substitutions_df.empty:
        return substitutions_df

    return substitutions_df.sort_values(
        ["opponent", "player_in", "player_off"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def normalize_substitution_reason(reason):
    reason = str(reason or "")
    if "Injury" in reason:
        return "Injury"
    return "Tactical"


def plot_starter_positions(starter_df, total_starts_df, formation_df, save_path):
    fig, ax = plt.subplots(figsize=(16, 11), facecolor="#0f1720")
    draw_pitch(ax, pitch_color="#14532d", line_color="#ecfdf5", alpha=0.95)

    used_positions = sorted(starter_df["position_id"].unique())
    for position_id in used_positions:
        info = POSITION_INFO[position_id]
        entries = starter_df[starter_df["position_id"] == position_id]
        players_block = "\n".join(
            f"{row.display_name} {int(row.starts_at_position)}"
            for row in entries.itertuples(index=False)
        )
        label = f"{info['abbr']}\n{players_block}"
        ax.text(
            info["xy"][0],
            info["xy"][1],
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#0f1720",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f8fafc",
                "edgecolor": "#991b1b",
                "linewidth": 1.5,
                "alpha": 0.95,
            },
        )

    top_starters = ", ".join(
        f"{row.display_name} ({int(row.total_starts)})"
        for row in total_starts_df.head(5).itertuples(index=False)
    )
    formation_summary = " | ".join(
        f"{row.formation_label}: {int(row.matches)}"
        for row in formation_df.itertuples(index=False)
    )

    ax.set_title(
        "Olympiacos Starters By Kickoff Position\n"
        f"Starting formations: {formation_summary}",
        color="white",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.03,
        f"Top starters: {top_starters}",
        ha="center",
        color="#e5e7eb",
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="#0f1720")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    match_rows = load_olympiacos_matches()
    starter_df, total_starts_df, formation_df = build_starter_position_summary(match_rows)
    substitutions_df = build_substitutions_table(match_rows)

    starter_csv = os.path.join(OUTPUT_DIR, "starter_positions.csv")
    total_starts_csv = os.path.join(OUTPUT_DIR, "player_total_starts.csv")
    formations_csv = os.path.join(OUTPUT_DIR, "starting_formations.csv")
    substitutions_csv = os.path.join(OUTPUT_DIR, "substitutions_table.csv")
    substitutions_md = os.path.join(OUTPUT_DIR, "substitutions_table.md")
    starters_plot = os.path.join(OUTPUT_DIR, "starter_positions.png")

    starter_df.to_csv(starter_csv, index=False)
    total_starts_df.to_csv(total_starts_csv, index=False)
    formation_df.to_csv(formations_csv, index=False)
    substitutions_df.to_csv(substitutions_csv, index=False)
    csv_to_markdown(substitutions_csv, substitutions_md)
    plot_starter_positions(starter_df, total_starts_df, formation_df, starters_plot)

    print(f"Olympiacos matches analysed: {len(match_rows)}")
    print(f"Saved: {starter_csv}")
    print(f"Saved: {total_starts_csv}")
    print(f"Saved: {formations_csv}")
    print(f"Saved: {substitutions_csv}")
    print(f"Saved: {substitutions_md}")
    print(f"Saved: {starters_plot}")


if __name__ == "__main__":
    main()
