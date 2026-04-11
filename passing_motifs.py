import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import os
import pandas as pd


def load_events(match_ids, zip_path):
    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for mid in match_ids:
            candidates = [n for n in names if n.endswith(f"{mid}.json")]
            if not candidates:
                continue
            with zf.open(candidates[0]) as f:
                for ev in json.load(f):
                    records.append((mid, ev))
    return records


def extract_pass_sequences(match_ids, team, zip_path):
    """
    Extracts all pass sequences for the team, grouped by possession.
    Returns a list of lists: [[player_A, player_B, player_C, ...], ...]
    """
    records = load_events(match_ids, zip_path)

    # group events by (match_id, possession)
    possessions = defaultdict(list)
    for mid, ev in records:
        if ev.get("team", {}).get("name") != team:
            continue
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77): # completed passes only
            continue
        passer = ev.get("player", {}).get("name")
        recipient = p.get("recipient", {}).get("name") if p.get("recipient") else None
        if not passer or not recipient:
            continue
        key = (mid, ev.get("possession", 0))
        possessions[key].append((passer, recipient))

    # reconstruct player chain per possession
    sequences = []
    for key, pass_list in possessions.items():
        if len(pass_list) < 2:
            continue
        # build passing chain
        chain = [pass_list[0][0], pass_list[0][1]]
        for i in range(1, len(pass_list)):
            chain.append(pass_list[i][1])
        sequences.append(chain)

    return sequences


def encode_motif(players):
    """
    Converts a player sequence into an abstract motif (A, B, C, D).
    Ex: [Messi, Xavi, Messi, Iniesta] -> ABAC
    """
    mapping = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    encoded = []
    for p in players:
        if p not in mapping:
            mapping[p] = letters[len(mapping)]
        encoded.append(mapping[p])
    return "".join(encoded)


MOTIFS_3 = ["ABAB", "ABAC", "ABCA", "ABCB", "ABCD"]
MOTIFS_GOAL = [m + "G" for m in MOTIFS_3]


def count_motifs(sequences, window=4, include_all=False):
    """
    Counts occurrences of each abstract motif in windows of `window` players.
    If include_all=True, counts ALL unique motifs, not just the standard ones.
    """
    counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - window + 1):
            motif = encode_motif(seq[i: i + window])
            counter[motif] += 1
    return counter


def plot_top_motifs_bar(match_ids, team, zip_path,
                        save_path="passing_motifs_bar.png", top_n=20):
    """
    Bar chart of the most frequent motifs across window sizes 2-4,
    including shot-ending variants.
    """
    sequences = extract_pass_sequences_with_shots(match_ids, team, zip_path)

    counter = Counter()
    for seq in sequences:
        for window in range(2, 5):
            for i in range(len(seq) - window + 1):
                slice_ = seq[i: i + window]
                if any(p == "GOAL" for p in slice_[:-1]):
                    continue
                players_only = [p for p in slice_ if p != "GOAL"]
                if len(players_only) < 2:
                    continue
                has_goal = slice_[-1] == "GOAL"
                motif    = encode_motif(players_only)
                if has_goal:
                    counter[motif + "G"] += 1
                else:
                    counter[motif] += 1

    top    = counter.most_common(top_n)
    labels = [t[0] for t in top]
    values = [t[1] for t in top]

    palette = {
        "AB":    "#888888",
        "ABA":   "#42a5f5",
        "ABC":   "#26c6da",
        "ABAB":  "#e53517",
        "ABAC":  "#42a5f5",
        "ABCA":  "#43a047",
        "ABCB":  "#ffca28",
        "ABCD":  "#ab47bc",
    }
    bar_colors = [palette.get(l, "#555555") for l in labels]

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(labels, values, color=bar_colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom",
            color="white", fontsize=9
        )

    ax.set_xlabel("motif", color="white", fontsize=11)
    ax.set_ylabel("frequency", color="white", fontsize=11)
    ax.set_title(
        f"{team} - top {top_n} passing motifs (window sizes 2-4, including shots)",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")
    ax.grid(axis="y", color="white", alpha=0.1, linestyle="--")

    legend_elements = [
        mpatches.Patch(facecolor=c, label=m)
        for m, c in palette.items()
    ]
    ax.legend(handles=legend_elements, facecolor="#0d0d1a", labelcolor="white",
              fontsize=9, framealpha=0.8, title="standard motifs",
              title_fontsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()

def get_top_players_by_minutes(match_ids, team, zip_path, top_n=5):
    """
    Returns top N players by approximate minutes played
    (number of distinct minutes in which they had at least one event).
    """
    records = load_events(match_ids, zip_path)
    minutes = defaultdict(set)
    for mid, ev in records:
        if ev.get("team", {}).get("name") != team:
            continue
        player = ev.get("player", {}).get("name")
        if not player:
            continue
        minutes[player].add((mid, ev.get("minute", 0)))

    sorted_players = sorted(minutes, key=lambda p: len(minutes[p]), reverse=True)
    return sorted_players[:top_n]


def extract_pass_sequences_with_shots(match_ids, team, zip_path):
    """
    Extracts pass sequences per possession, appending "GOAL" sentinel
    at the end if the possession ended in a shot.
    Returns list of player chains: [..., "GOAL"] if ended in shot.
    """
    records = load_events(match_ids, zip_path)

    # build shot set: possession keys that ended in a shot
    shot_possessions = set()
    for mid, ev in records:
        if ev.get("team", {}).get("name") != team:
            continue
        if ev.get("type", {}).get("id") == 16:  # Shot
            key = (mid, ev.get("possession", 0))
            shot_possessions.add(key)

    # group passes by possession
    possessions = defaultdict(list)
    for mid, ev in records:
        if ev.get("team", {}).get("name") != team:
            continue
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77):
            continue
        passer = ev.get("player", {}).get("name")
        recipient = p.get("recipient", {}).get("name") if p.get("recipient") else None
        if not passer or not recipient:
            continue
        key = (mid, ev.get("possession", 0))
        possessions[key].append((passer, recipient))

    sequences = []
    for key, pass_list in possessions.items():
        if len(pass_list) < 2:
            continue
        chain = [pass_list[0][0], pass_list[0][1]]
        for i in range(1, len(pass_list)):
            chain.append(pass_list[i][1])
        if key in shot_possessions:
            chain.append("GOAL")
        sequences.append(chain)

    return sequences


def count_motifs_for_player(sequences, player, window=4):
    """
    Counts motifs in windows of `window` players where the target player
    appears at least once. The motif is encoded relative to player positions,
    with the target player always mapped to a consistent letter.
    Also counts motifs ending in GOAL sentinel.
    Returns two Counters: (standard_motifs, goal_motifs)
    """
    standard = Counter()
    goal = Counter()

    for seq in sequences:
        if player not in seq:
            continue

        for i in range(len(seq) - window + 1):
            window_slice = seq[i: i + window]

            if "GOAL" in window_slice[:-1]:
                continue

            if player not in window_slice:
                continue

            has_goal = window_slice[-1] == "GOAL"
            players_only = [p for p in window_slice if p != "GOAL"]

            if len(players_only) < window - (1 if has_goal else 0):
                continue

            motif = encode_motif(players_only)
            if has_goal:
                goal[motif + "G"] += 1
            else:
                standard[motif] += 1

    return standard, goal

def count_motifs_for_player_extended(sequences, player):
    """
    Counts motifs of window sizes 2, 3, 4 (players only),
    plus shot-ending variants by detecting GOAL sentinel.
    Matches the Bekker & Dabadghao approach.
    """
    standard = Counter()
    goal = Counter()

    for seq in sequences:
        if player not in seq:
            continue

        for window in range(2, 5):  # windows of 2, 3, 4 players
            for i in range(len(seq) - window + 1):
                slice_ = seq[i: i + window]

                # skip if GOAL appears in middle (not at end)
                if any(p == "GOAL" for p in slice_[:-1]):
                    continue

                players_only = [p for p in slice_ if p != "GOAL"]
                if player not in players_only:
                    continue
                if len(players_only) < 2:
                    continue

                has_goal = slice_[-1] == "GOAL"
                motif = encode_motif(players_only)

                if has_goal:
                    goal[motif + "G"] += 1
                else:
                    standard[motif] += 1

    return standard, goal


MOTIFS_EXTENDED = [
    "AB",
    "ABA", "ABC",
    "ABAB", "ABAC", "ABCA", "ABCB", "ABCD",
    "ABG", "ABAG", "ABCG", "ABABG", "ABACG", "ABCAG", "ABCBG", "ABCDG",
]

MOTIF_EXPLANATIONS = {
    "AB":    "simple pass between 2 players",
    "ABA":   "pass and return (wall pass)",
    "ABC":   "2 passes, 3 different players",
    "ABAB":  "double triangle",
    "ABAC":  "central pivot, different target",
    "ABCA":  "3-player circuit back to initiator",
    "ABCB":  "B is distribution pivot",
    "ABCD":  "direct, 4 different players",
    "ABG":   "1 pass → shot",
    "ABAG":  "wall pass → shot",
    "ABCG":  "2 passes → shot",
    "ABABG": "double triangle → shot",
    "ABACG": "central pivot → shot",
    "ABCAG": "3-player circuit → shot",
    "ABCBG": "B pivot → shot",
    "ABCDG": "direct sequence → shot",
}


def plot_player_motif_radar(match_ids, team, zip_path,
                            top_n=5,
                            save_dir="output/"):
    """
    Draws one radar chart per player (top N by minutes played).
    Normalizes all motif counts against the global maximum across all players,
    following the Bekker & Dabadghao approach.
    """
    top_players = get_top_players_by_minutes(match_ids, team, zip_path, top_n=top_n)
    sequences   = extract_pass_sequences_with_shots(match_ids, team, zip_path)

    print(f"Top {top_n} players by minutes: {top_players}")

    all_counts = {}
    for player in top_players:
        standard, goal = count_motifs_for_player_extended(sequences, player)
        combined = {}
        for m in MOTIFS_EXTENDED:
            if m.endswith("G"):
                combined[m] = goal.get(m, 0)
            else:
                combined[m] = standard.get(m, 0)
        all_counts[player] = {
            "combined": combined,
            "standard": standard,
            "goal": goal,
        }

    global_max = max(
        v
        for player_data in all_counts.values()
        for v in player_data["combined"].values()
    )
    if global_max == 0:
        global_max = 1

    N = len(MOTIFS_EXTENDED)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for player in top_players:
        combined = all_counts[player]["combined"]
        standard = all_counts[player]["standard"]
        goal     = all_counts[player]["goal"]

        # normalize against global max
        values = [combined[m] / global_max for m in MOTIFS_EXTENDED]
        values += values[:1]

        # separate standard vs goal for coloring
        std_values = [combined[m] / global_max if not m.endswith("G") else 0
                       for m in MOTIFS_EXTENDED] + [0]
        goal_values = [combined[m] / global_max if m.endswith("G") else 0
                       for m in MOTIFS_EXTENDED] + [0]

        fig, ax = plt.subplots(
            figsize=(9, 9), subplot_kw={"projection": "polar"},
            facecolor="#1a1a2e"
        )
        ax.set_facecolor("#1a1a2e")

        ax.plot(angles, std_values, color="#e53517", lw=2,
                label="standard motifs", alpha=0.9)
        ax.fill(angles, std_values, color="#e53517", alpha=0.15)

        ax.plot(angles, goal_values, color="#ffca28", lw=2,
                label="motifs ending in shot", alpha=0.9)
        ax.fill(angles, goal_values, color="#ffca28", alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            MOTIFS_EXTENDED, color="white", fontsize=9, fontweight="bold"
        )
        ax.set_yticklabels([])
        ax.grid(color="white", alpha=0.15, linestyle="--")
        ax.spines["polar"].set_color("white")
        ax.spines["polar"].set_alpha(0.3)

        total_std  = sum(standard.values())
        total_goal = sum(goal.values())

        ax.set_title(
            f"{player}\n"
            f"standard: {total_std} motifs  ·  shot-ending: {total_goal} motifs\n"
            f"normalized against global max ({global_max})",
            color="white", fontsize=12, fontweight="bold", pad=20
        )

        ax.legend(
            loc="upper right", bbox_to_anchor=(1.4, 1.15),
            facecolor="#0d0d1a", labelcolor="white", fontsize=10, framealpha=0.8
        )

        legend_text = "\n".join(
            f"{k}: {v}" for k, v in MOTIF_EXPLANATIONS.items()
        )
        fig.text(
            0.5, 0.01, legend_text,
            ha="center", va="bottom", color="#b0b0b0",
            fontsize=7, fontstyle="italic",
            bbox=dict(facecolor="#0d0d1a", alpha=0.7, edgecolor="none", pad=5)
        )

        plt.tight_layout(rect=[0, 0.22, 1, 1])

        safe_name = player.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(save_dir, f"motif_radar_{safe_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"Saved: {save_path}")
        plt.close()


def encode_motif_player_centric(players, target_player):
    """
    Encodes motif with target_player always mapped to 'A'.
    Other players get B, C, D in order of appearance.
    This makes motifs meaningful per player:
      ABA = target receives, passes, receives again (pivot)
      ABC = target passes to two different players
      BAC = target is in the middle of an exchange
      BAB = target receives from B and passes back to B
    """
    mapping = {target_player: "A"}
    letters = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = 0
    encoded = []
    for p in players:
        if p not in mapping:
            mapping[p] = letters[idx]
            idx += 1
        encoded.append(mapping[p])
    return "".join(encoded)


def count_motifs_player_centric(sequences, player):
    """
    Counts player-centric motifs across window sizes 2-4,
    including shot-ending variants.
    Target player is always encoded as A.
    """
    standard = Counter()
    goal = Counter()

    for seq in sequences:
        if player not in seq:
            continue

        for window in range(2, 5):
            for i in range(len(seq) - window + 1):
                slice_ = seq[i: i + window]

                if any(p == "GOAL" for p in slice_[:-1]):
                    continue

                players_only = [p for p in slice_ if p != "GOAL"]
                if player not in players_only:
                    continue
                if len(players_only) < 2:
                    continue

                has_goal = slice_[-1] == "GOAL"
                motif = encode_motif_player_centric(players_only, player)

                if has_goal:
                    goal[motif + "G"] += 1
                else:
                    standard[motif] += 1

    return standard, goal


# player-centric motif labels - A is always the target player
MOTIFS_CENTRIC = [
    # 1 pass: A passes or receives
    "AB",   # A passes to B
    "BA",   # A receives from B
    # 2 passes: A involved in 3-player sequence
    "ABA",  # A passes to B, gets it back
    "ABА",  # same as above (covered)
    "ABC",  # A passes to B, B passes to C
    "BAC",  # B passes to A, A passes to C
    "BAB",  # B passes to A, A passes back to B
    "BАC",  # same
    # 3 passes: A in 4-player sequence
    "ABAB", # A-B exchange twice
    "ABAC", # A passes to B then C
    "ABCA", # A→B→C→A circuit
    "BABC", # B→A→B→C
    "BACD", # B→A→C→D (A as distributor)
    "BACA", # B→A→C→A
    # shot-ending
    "ABG",  # A passes, shot follows
    "BAG",  # A receives, shot follows
    "ABCG", # A passes, 2 more passes, shot
    "BACG", # A receives, passes, shot
    "ABACG","BABCG",
]

# deduplicate while preserving order
seen = set()
MOTIFS_CENTRIC = [m for m in MOTIFS_CENTRIC
                  if not (m in seen or seen.add(m))]

MOTIF_CENTRIC_EXPLANATIONS = {
    "AB":    "A passes to B",
    "BA":    "A receives from B",
    "ABA":   "A passes to B, gets it back (wall pass)",
    "ABC":   "A passes to B, B passes to C",
    "BAC":   "A receives from B, passes to C (distributor)",
    "BAB":   "A receives from B, passes back to B",
    "ABAB":  "A-B exchange twice",
    "ABAC":  "A passes to B then to C",
    "ABCA":  "A→B→C circuit returns to A",
    "BABC":  "A receives, passes to B, B passes to C",
    "BACD":  "A receives, passes to C, C passes to D",
    "BACA":  "A receives, passes to C, gets it back",
    "ABG":   "A passes → shot",
    "BAG":   "A receives → shot",
    "ABCG":  "A passes, 2 more passes → shot",
    "BACG":  "A receives, passes → shot",
    "ABACG": "A passes to B then C → shot",
    "BABCG": "A receives, passes, sequence → shot",
}


def plot_player_motif_radar_centric(match_ids, team, zip_path,
                                    top_n=5,
                                    save_dir="output/"):
    """
    Draws player-centric radar charts (top N by minutes played).
    Target player is always A - motifs show their specific passing role.
    Saves as motif_radar_centric_{player_name}.png
    """
    top_players = get_top_players_by_minutes(match_ids, team, zip_path, top_n=top_n)
    sequences = extract_pass_sequences_with_shots(match_ids, team, zip_path)

    print(f"Top {top_n} players (player-centric): {top_players}")

    all_counts = {}
    for player in top_players:
        standard, goal = count_motifs_player_centric(sequences, player)
        combined = {}
        for m in MOTIFS_CENTRIC:
            if m.endswith("G"):
                combined[m] = goal.get(m, 0)
            else:
                combined[m] = standard.get(m, 0)
        all_counts[player] = {
            "combined": combined,
            "standard": standard,
            "goal": goal,
        }

    global_max = max(
        v
        for player_data in all_counts.values()
        for v in player_data["combined"].values()
    )
    if global_max == 0:
        global_max = 1

    N = len(MOTIFS_CENTRIC)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for player in top_players:
        combined = all_counts[player]["combined"]
        standard = all_counts[player]["standard"]
        goal = all_counts[player]["goal"]

        std_values = [combined[m] / global_max if not m.endswith("G") else 0
                       for m in MOTIFS_CENTRIC] + [0]
        goal_values = [combined[m] / global_max if m.endswith("G") else 0
                       for m in MOTIFS_CENTRIC] + [0]

        fig, ax = plt.subplots(
            figsize=(9, 9), subplot_kw={"projection": "polar"},
            facecolor="#1a1a2e"
        )
        ax.set_facecolor("#1a1a2e")

        ax.plot(angles, std_values, color="#e53517", lw=2,
                label="standard motifs (A = this player)", alpha=0.9)
        ax.fill(angles, std_values, color="#e53517", alpha=0.15)

        ax.plot(angles, goal_values, color="#ffca28", lw=2,
                label="motifs ending in shot", alpha=0.9)
        ax.fill(angles, goal_values, color="#ffca28", alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            MOTIFS_CENTRIC, color="white", fontsize=9, fontweight="bold"
        )
        ax.set_yticklabels([])
        ax.grid(color="white", alpha=0.15, linestyle="--")
        ax.spines["polar"].set_color("white")
        ax.spines["polar"].set_alpha(0.3)

        total_std = sum(standard.values())
        total_goal = sum(goal.values())

        ax.set_title(
            f"{player} - player-centric motifs\n"
            f"standard: {total_std}  ·  shot-ending: {total_goal}\n"
            f"normalized against global max ({global_max})",
            color="white", fontsize=11, fontweight="bold", pad=20
        )

        ax.legend(
            loc="upper right", bbox_to_anchor=(1.45, 1.15),
            facecolor="#0d0d1a", labelcolor="white", fontsize=9, framealpha=0.8
        )

        legend_text = "\n".join(
            f"{k}: {v}" for k, v in MOTIF_CENTRIC_EXPLANATIONS.items()
        )
        fig.text(
            0.5, 0.01, legend_text,
            ha="center", va="bottom", color="#b0b0b0",
            fontsize=7, fontstyle="italic",
            bbox=dict(facecolor="#0d0d1a", alpha=0.7, edgecolor="none", pad=5)
        )

        plt.tight_layout(rect=[0, 0.25, 1, 1])

        safe_name = player.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(save_dir, f"motif_radar_centric_{safe_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"Saved: {save_path}")
        plt.close()



if __name__ == "__main__":
    TEAM_MATCHES_CSV  = "Olympiacos Piraeus"
    TEAM_JSON = "Olympiacos"
    ZIP_PATH  = "data/statsbomb/league_phase.zip"
    MATCHES_CSV = "data/matches.csv"
    OUT = "output/"
    os.makedirs(OUT, exist_ok=True)

    matches_df = pd.read_csv(MATCHES_CSV)

    # filter available matches
    with zipfile.ZipFile(ZIP_PATH) as zf:
        available = set(zf.namelist())

    all_ids = matches_df[
        (matches_df["home"] == TEAM_MATCHES_CSV) | (matches_df["away"] == TEAM_MATCHES_CSV)
    ]["statsbomb"].tolist()
    all_ids = [mid for mid in all_ids if f"{mid}.json" in available]

    home_ids = matches_df[matches_df["home"] == TEAM_MATCHES_CSV]["statsbomb"].tolist()
    home_ids = [mid for mid in home_ids if f"{mid}.json" in available]

    away_ids = matches_df[matches_df["away"] == TEAM_MATCHES_CSV]["statsbomb"].tolist()
    away_ids = [mid for mid in away_ids if f"{mid}.json" in available]

    print(f"{TEAM_MATCHES_CSV}: {len(all_ids)} available matches ({len(home_ids)} home, {len(away_ids)} away)")

    plot_player_motif_radar(
        all_ids, TEAM_JSON, ZIP_PATH,
        top_n=5,
        save_dir=OUT
    )

    plot_player_motif_radar_centric(
        all_ids, TEAM_JSON, ZIP_PATH,
        top_n=5,
        save_dir=OUT
    )

    plot_top_motifs_bar(
        all_ids, TEAM_JSON, ZIP_PATH,
        OUT + "passing_motifs_bar.png", top_n=20
    )