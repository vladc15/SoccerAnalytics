import json
import os
import zipfile
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


TEAM_MATCHES_CSV = "Olympiacos Piraeus"
TEAM_STATSBOMB = "Olympiacos"
ZIP_PATHS = [
    "data/statsbomb/league_phase.zip",
    "data/statsbomb/playoffs.zip",
]
MATCHES_CSV = "data/matches.csv"
POSITION_ABBREVIATIONS = {
    1: "GK",
    2: "RB",
    3: "RCB",
    4: "CB",
    5: "LCB",
    6: "LB",
    7: "RWB",
    8: "LWB",
    9: "RDM",
    10: "CDM",
    11: "LDM",
    12: "RM",
    13: "RCM",
    14: "CM",
    15: "LCM",
    16: "LM",
    17: "RW",
    18: "RAM",
    19: "CAM",
    20: "LAM",
    21: "LW",
    22: "RCF",
    23: "ST",
    24: "LCF",
    25: "SS",
}


def load_events(match_ids, zip_paths):
    records = []
    for one_zip_path in zip_paths:
        with zipfile.ZipFile(one_zip_path, "r") as zf:
            names = zf.namelist()
            for mid in match_ids:
                candidates = [name for name in names if name.endswith(f"{mid}.json")]
                if not candidates:
                    continue
                with zf.open(candidates[0]) as f:
                    for ev in json.load(f):
                        records.append((mid, ev))
    return records


def team_events(records, team):
    return [(mid, ev) for mid, ev in records if ev.get("team", {}).get("name") == team]


def load_match_events(match_ids, zip_paths):
    events_by_match = {}
    for one_zip_path in zip_paths:
        with zipfile.ZipFile(one_zip_path, "r") as zf:
            names = set(zf.namelist())
            for mid in match_ids:
                name = f"{mid}.json"
                if name not in names:
                    continue
                with zf.open(name) as f:
                    events_by_match[str(mid)] = json.load(f)
    return events_by_match


def load_single_match_events(match_id, zip_paths):
    events_by_match = load_match_events([str(match_id)], zip_paths)
    events = events_by_match.get(str(match_id))
    if events is None:
        raise ValueError(f"{match_id}.json not found in {zip_paths}")
    return events


def competition_match_ids(zip_paths):
    match_ids = set()
    for one_zip_path in zip_paths:
        with zipfile.ZipFile(one_zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json") or name.endswith("_lineups.json"):
                    continue
                match_ids.add(os.path.basename(name).replace(".json", ""))
    return sorted(match_ids)


def most_common_value(counter_obj):
    if not counter_obj:
        return None
    return sorted(counter_obj.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def iter_lineup_entries(zip_paths, match_ids=None):
    allowed_match_ids = set(str(mid) for mid in match_ids) if match_ids is not None else None

    for one_zip_path in zip_paths:
        with zipfile.ZipFile(one_zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith("_lineups.json"):
                    continue
                match_id = os.path.basename(name).replace("_lineups.json", "")
                if allowed_match_ids is not None and match_id not in allowed_match_ids:
                    continue
                with zf.open(name) as f:
                    payload = json.load(f)
                for team_entry in payload:
                    yield match_id, team_entry


def competition_player_profiles(match_ids, zip_paths):
    profiles = defaultdict(lambda: {
        "display_name": None,
        "team_counts": Counter(),
        "appearance_position_counts": Counter(),
        "starting_position_counts": Counter(),
        "appearances": 0,
        "starts": 0,
    })

    for _match_id, team_entry in iter_lineup_entries(zip_paths, match_ids=match_ids):
        team_name = team_entry.get("team_name")
        for player in team_entry.get("lineup", []):
            player_id = player.get("player_id")
            player_name_value = player.get("player_name")
            if player_id is None or not player_name_value:
                continue

            positions = player.get("positions", [])
            if not positions:
                continue

            first_position = positions[0]
            position_id = first_position.get("position_id")

            profile = profiles[(player_id, player_name_value)]
            nickname = player.get("player_nickname")
            profile["display_name"] = (
                nickname.strip()
                if isinstance(nickname, str) and nickname.strip()
                else player_name_value
            )
            profile["team_counts"][team_name] += 1
            profile["appearances"] += 1

            if position_id is not None:
                profile["appearance_position_counts"][position_id] += 1

            if first_position.get("start_reason") == "Starting XI":
                profile["starts"] += 1
                if position_id is not None:
                    profile["starting_position_counts"][position_id] += 1

    rows = []
    for (player_id, player_name_value), profile in profiles.items():
        primary_position_id = most_common_value(profile["appearance_position_counts"])
        primary_starting_position_id = most_common_value(profile["starting_position_counts"])
        primary_team = most_common_value(profile["team_counts"])

        rows.append({
            "player_id": player_id,
            "player": player_name_value,
            "display_name": profile["display_name"],
            "team_profile": primary_team,
            "appearances": profile["appearances"],
            "starts": profile["starts"],
            "primary_position_id": primary_position_id,
            "primary_position_abbreviation": POSITION_ABBREVIATIONS.get(primary_position_id),
            "primary_starting_position_id": primary_starting_position_id,
            "primary_starting_position_abbreviation": POSITION_ABBREVIATIONS.get(primary_starting_position_id),
        })

    return pd.DataFrame(rows)


def load_lineup_payload(statsbomb_match_id, zip_paths=ZIP_PATHS):
    lineup_file = f"{statsbomb_match_id}_lineups.json"
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if lineup_file not in zf.namelist():
                continue
            with zf.open(lineup_file) as f:
                return json.load(f)
    raise FileNotFoundError(f"{lineup_file} not found in {zip_paths}")


def team_lineup_entry(statsbomb_match_id, team_name, zip_paths=ZIP_PATHS):
    payload = load_lineup_payload(statsbomb_match_id, zip_paths=zip_paths)
    for team in payload:
        if team.get("team_name") == team_name:
            return team
    raise ValueError(f"{team_name} not found in lineup file for match {statsbomb_match_id}")


def csv_to_markdown(csv_path, output_path=None):
    df = pd.read_csv(csv_path)
    md = df.to_markdown(index=False)

    if output_path:
        with open(output_path, "w") as f:
            f.write(md)
        print(f"Saved: {output_path}")
    else:
        print(md)


def draw_pitch(ax, pitch_color="#1a1a2e", line_color="white", alpha=0.9):
    ax.set_facecolor(pitch_color)
    lw = 1.5

    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=line_color, lw=lw, alpha=alpha)

    def rect(x, y, w, h):
        ax.add_patch(
            mpatches.Rectangle(
                (x, y), w, h,
                fill=False, edgecolor=line_color, lw=lw, alpha=alpha,
            )
        )

    rect(0, 0, 120, 80)
    line(60, 0, 60, 80)
    circle = plt.Circle((60, 40), 10, fill=False, color=line_color, lw=lw, alpha=alpha)
    ax.add_patch(circle)
    ax.plot(60, 40, "o", color=line_color, ms=2, alpha=alpha)

    rect(0, 18, 18, 44)
    rect(102, 18, 18, 44)
    rect(0, 30, 6, 20)
    rect(114, 30, 6, 20)
    ax.plot(12, 40, "o", color=line_color, ms=3, alpha=alpha)
    ax.plot(108, 40, "o", color=line_color, ms=3, alpha=alpha)
    arc_l = mpatches.Arc((12, 40), 20, 20, angle=0, theta1=308, theta2=52, color=line_color, lw=lw, alpha=alpha)
    arc_r = mpatches.Arc((108, 40), 20, 20, angle=0, theta1=128, theta2=232, color=line_color, lw=lw, alpha=alpha)
    ax.add_patch(arc_l)
    ax.add_patch(arc_r)

    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def make_heatmap(coords, sigma=3, bins=(120, 80)):
    if not coords:
        return np.zeros((bins[1], bins[0]))

    heatmap, _, _ = np.histogram2d(
        [coord[0] for coord in coords],
        [coord[1] for coord in coords],
        bins=bins,
        range=[[0, 120], [0, 80]],
    )
    return gaussian_filter(heatmap.T, sigma=sigma)
