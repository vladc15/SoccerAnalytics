import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import networkx as nx
import os


def draw_pitch(ax, pitch_color="#1a1a2e", line_color="white", alpha=0.9):
    ax.set_facecolor(pitch_color)
    lw = 1.5

    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=line_color, lw=lw, alpha=alpha)

    def rect(x, y, w, h):
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h,
            fill=False, edgecolor=line_color, lw=lw, alpha=alpha
        ))

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
    arc_l = mpatches.Arc((12, 40), 20, 20, angle=0, theta1=308, theta2=52,
                          color=line_color, lw=lw, alpha=alpha)
    arc_r = mpatches.Arc((108, 40), 20, 20, angle=0, theta1=128, theta2=232,
                          color=line_color, lw=lw, alpha=alpha)
    ax.add_patch(arc_l)
    ax.add_patch(arc_r)

    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax



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


def team_events(records, team):
    return [(mid, ev) for mid, ev in records if ev.get("team", {}).get("name") == team]


def extract_locations(records, team, type_ids):
    """Extract coords (x, y) for events in type_ids."""
    coords = []
    for _mid, ev in records:
        if ev.get("team", {}).get("name") != team:
            continue
        if ev.get("type", {}).get("id") not in type_ids:
            continue
        loc = ev.get("location")
        if loc:
            coords.append((loc[0], loc[1]))
    return coords



def make_heatmap(coords, sigma=3, bins=(120, 80)):
    if not coords:
        return np.zeros((bins[1], bins[0]))

    heatmap, _, _ = np.histogram2d(
        [c[0] for c in coords],
        [c[1] for c in coords],
        bins=bins,
        range=[[0, 120], [0, 80]],
    )
    heatmap = heatmap.T
    return gaussian_filter(heatmap, sigma=sigma)


CMAP_RED   = LinearSegmentedColormap.from_list("red_hm",   ["#1a1a2e", "#6a0f2a", "#c0392b", "#e74c3c", "#ff8a80"])
CMAP_BLUE  = LinearSegmentedColormap.from_list("blue_hm",  ["#1a1a2e", "#0d2b5e", "#1565c0", "#42a5f5", "#b3e5fc"])
CMAP_GREEN = LinearSegmentedColormap.from_list("green_hm", ["#1a1a2e", "#0a3d1f", "#1b5e20", "#43a047", "#b9f6ca"])
CMAP_GOLD  = LinearSegmentedColormap.from_list("gold_hm",  ["#1a1a2e", "#4a2c00", "#bf8000", "#ffca28", "#fff9c4"])


def plot_error_heatmap(match_ids, team, zip_path, save_path="heatmap_errors.png"):
    records = load_events(match_ids, zip_path)
    coords = extract_locations(records, team, type_ids={38, 3})

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a1a2e")
    draw_pitch(ax)

    hm = make_heatmap(coords, sigma=4)
    ax.imshow(
        hm, extent=[0, 120, 0, 80], origin="lower",
        cmap=CMAP_RED, alpha=0.75, aspect="auto"
    )

    if coords:
        xs, ys = zip(*coords)
        ax.scatter(xs, ys, s=18, color="white", alpha=0.35, linewidths=0)

    ax.set_title(
        f"{team} - errors on field (miscontrol + dispossessed)\n"
        f"n={len(coords)} events · {len(match_ids)} matches",
        color="white", fontsize=13, pad=12, fontweight="bold"
    )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=CMAP_RED, norm=plt.Normalize(vmin=hm.min(), vmax=hm.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("density", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()



def plot_shot_heatmap(match_ids, team, zip_path, save_path="heatmap_shots.png"):
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    coords_all, coords_goal = [], []
    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 16:
            continue
        loc = ev.get("location")
        if not loc:
            continue
        coords_all.append((loc[0], loc[1]))
        if ev.get("shot", {}).get("outcome", {}).get("id") == 97:
            coords_goal.append((loc[0], loc[1]))

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a1a2e")
    draw_pitch(ax)

    hm = make_heatmap(coords_all, sigma=3)
    ax.imshow(
        hm, extent=[0, 120, 0, 80], origin="lower",
        cmap=CMAP_GOLD, alpha=0.7, aspect="auto"
    )

    if coords_all:
        xs, ys = zip(*coords_all)
        ax.scatter(xs, ys, s=22, color="white", alpha=0.25, linewidths=0)

    if coords_goal:
        gxs, gys = zip(*coords_goal)
        ax.scatter(
            gxs, gys, s=80, color="#ffca28", alpha=0.9,
            edgecolors="white", linewidths=0.8, marker="*"
        )

    legend_elements = [
        mpatches.Patch(facecolor="#ffca28", label=f"Goals ({len(coords_goal)})"),
        mpatches.Patch(facecolor="white", alpha=0.4, label=f"Shots ({len(coords_all)})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", facecolor="#1a1a2e",
              labelcolor="white", fontsize=10, framealpha=0.8)

    ax.set_title(
        f"{team} - shots on field\n"
        f"* = goal · {len(match_ids)} matches",
        color="white", fontsize=13, pad=12, fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()



def plot_pressure_heatmap(match_ids, team, zip_path, save_path="heatmap_pressure.png"):
    """
    Received pressure = events of type Pressure (id 17) by opponent team that occur in Olympiacos half of the pitch.
    """
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

    # flip x: adversary attacks right-to-left relative to Olympiacos, so flip to show from Olympiacos perspective
    # depends on convention - if you want from Olympiacos perspective, flip:
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



def plot_pass_heatmap(match_ids, team, zip_path, save_path="heatmap_passes.png"):
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    origins, destinations = [], []
    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77):
            continue
        loc = ev.get("location")
        end = p.get("end_location")
        if loc:
            origins.append((loc[0], loc[1]))
        if end:
            destinations.append((end[0], end[1]))

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), facecolor="#1a1a2e")
    fig.suptitle(
        f"{team} - completed passes: origin vs destination  ({len(match_ids)} matches)",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )

    for ax, coords, title, cmap in [
        (axes[0], origins, f"Origin ({len(origins)} passes)", CMAP_GREEN),
        (axes[1], destinations, f"Destination", CMAP_GOLD),
    ]:
        draw_pitch(ax)
        hm = make_heatmap(coords, sigma=4)
        ax.imshow(
            hm, extent=[0, 120, 0, 80], origin="lower",
            cmap=cmap, alpha=0.75, aspect="auto"
        )
        ax.set_title(title, color="white", fontsize=12, pad=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()



def normalize_direction(coords_with_period):
    """
    Flips coordinates so the team always attacks right (x > 60).
    Based on average x in period 1: if > 60, team attacks left in period 1
    and coordinates need to be flipped.

    coords_with_period: list of (x, y, period)
    Returns list of (x, y) normalized.
    """
    p1 = [(x, y) for x, y, period in coords_with_period if period == 1]
    if not p1:
        return [(x, y) for x, y, _ in coords_with_period]

    avg_x_p1 = sum(x for x, y in p1) / len(p1)
    flip_p1 = avg_x_p1 > 60

    result = []
    for x, y, period in coords_with_period:
        if (period == 1 and flip_p1) or (period == 2 and not flip_p1):
            result.append((120 - x, 80 - y))
        else:
            result.append((x, y))
    return result

def get_attack_direction(events):
    """
    Determines attack direction for a match by looking at the team's
    average x position of passes in period 1.
    Returns True if coordinates need to be flipped (team attacks left in period 1).
    """
    p1_x = [
        ev["location"][0]
        for ev in events
        if ev.get("type", {}).get("id") == 30
        and ev.get("period") == 1
        and ev.get("location")
    ]
    if not p1_x:
        return False
    return np.mean(p1_x) > 60

def plot_passing_network(match_ids, team, zip_path,
                         save_path="passing_network.png",
                         min_passes=10, top_n_players=16):
    """
    Pass graph with nodes positioned on the pitch (average position of player when passing)
    and edges weighted by pass count.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    edges = defaultdict(lambda: {"count": 0, "obv": 0.0})
    player_positions = defaultdict(list)

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77):
            continue
        passer = ev.get("player", {}).get("name")
        recipient = p.get("recipient", {}).get("name") if p.get("recipient") else None
        loc = ev.get("location")
        if not passer or not recipient or not loc:
            continue

        key = (passer, recipient)
        edges[key]["count"] += 1
        edges[key]["obv"] += ev.get("obv_total_net") or 0.0
        player_positions[passer].append((loc[0], loc[1], ev.get("period", 1)))

    # (not) replaced above with this flipping
    # with zipfile.ZipFile(zip_path, "r") as zf:
    #     names = zf.namelist()
    #     for mid in match_ids:
    #         candidates = [n for n in names if n.endswith(f"{mid}.json")]
    #         if not candidates:
    #             continue
    #         with zf.open(candidates[0]) as f:
    #             events = json.load(f)

    #         # determine flip once per match
    #         team_pass_events = [
    #             ev for ev in events
    #             if ev.get("team", {}).get("name") == team
    #             and ev.get("type", {}).get("id") == 30
    #         ]
    #         flip = get_attack_direction(team_pass_events)

    #         for ev in team_pass_events:
    #             p = ev.get("pass", {})
    #             if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77):
    #                 continue
    #             passer    = ev.get("player", {}).get("name")
    #             recipient = p.get("recipient", {}).get("name") if p.get("recipient") else None
    #             loc       = ev.get("location")
    #             if not passer or not recipient or not loc:
    #                 continue

    #             x, y = loc[0], loc[1]
    #             if flip:
    #                 x, y = 120 - x, 80 - y

    #             key = (passer, recipient)
    #             edges[key]["count"] += 1
    #             edges[key]["obv"]   += ev.get("obv_total_net") or 0.0
    #             player_positions[passer].append((x, y))

    # top N players by total pass volume (in or out)
    player_volume = defaultdict(int)
    for (p, r), d in edges.items():
        player_volume[p] += d["count"]
        player_volume[r] += d["count"]
    top_players = set(
        sorted(player_volume, key=player_volume.get, reverse=True)[:top_n_players]
    )

    G = nx.DiGraph()
    for (passer, recipient), data in edges.items():
        if data["count"] < min_passes:
            continue
        if passer not in top_players or recipient not in top_players:
            continue
        G.add_edge(passer, recipient, weight=data["count"],
                   avg_obv=data["obv"] / data["count"])

    # node position = average location of player when passing
    pos = {}
    for player in G.nodes():
        locs = player_positions.get(player, [])
        if locs:
            # normalized = normalize_direction(locs)
            # pos[player] = (np.mean([l[0] for l in normalized]),
                        #    np.mean([l[1] for l in normalized]))
            pos[player] = (np.mean([l[0] for l in locs]),
                           np.mean([l[1] for l in locs]))
        else:
            pos[player] = (60, 40)

    # edge width
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [w / max_w * 8 for w in weights]

    # edge color: green = positive OBV, red = negative
    edge_colors = []
    for u, v in G.edges():
        obv = G[u][v]["avg_obv"]
        edge_colors.append("#43a047" if obv >= 0 else "#e53935")

    # node size = volume of passes
    node_sizes = []
    for node in G.nodes():
        vol = player_volume.get(node, 1)
        node_sizes.append(max(300, vol * 4))

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#1a1a2e")
    draw_pitch(ax, pitch_color="#1a1a2e")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.65,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color="#e53517",
        edgecolors="white",
        linewidths=1.5,
        alpha=0.95,
    )

    labels = {n: n.split()[-1] for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8, font_color="white", font_weight="bold",
    )

    legend_elements = [
        mpatches.Patch(facecolor="#43a047", label="pass with positive OBV"),
        mpatches.Patch(facecolor="#e53935", label="pass with negative OBV"),
        mpatches.Patch(facecolor="#e53517", edgecolor="white", label="player (size = pass volume)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", facecolor="#0d0d1a",
              labelcolor="white", fontsize=9, framealpha=0.85)

    ax.set_title(
        f"{team} - passing network  (min {min_passes} passes per pair)\n"
        f"{len(match_ids)} matches · edge width = pass volume",
        color="white", fontsize=13, pad=12, fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()



def plot_pass_clusters(match_ids, team, zip_path,
                       save_path="output/pass_clusters.png"):
    """
    Bar chart showing the distribution of pass clusters for the team.
    Uses pass_cluster_label and pass_cluster_id from StatsBomb pass events.
    Clusters are sorted by frequency.
    """
    records = load_events(match_ids, zip_path)
    team_ev = team_events(records, team)

    cluster_counts = defaultdict(lambda: {"count": 0, "label": ""})

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        cid    = p.get("pass_cluster_id")
        clabel = p.get("pass_cluster_label", "")
        if cid is None:
            continue
        cluster_counts[cid]["count"] += 1
        if clabel:
            cluster_counts[cid]["label"] = clabel

    if not cluster_counts:
        print("No cluster data found - check if pass_cluster_id is present in events.")
        return

    total = sum(v["count"] for v in cluster_counts.values())
    rows  = sorted(cluster_counts.items(), key=lambda x: x[1]["count"], reverse=True)

    labels = [f"C{cid}\n{v['label'][:20]}" for cid, v in rows]
    values = [v["count"] for _, v in rows]
    pcts   = [round(v["count"] / total * 100, 1) for _, v in rows]

    fig, ax = plt.subplots(figsize=(20, 7), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(values)))[::-1]

    bars = ax.bar(range(len(values)), values, color=colors,
                  edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3)

    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct}%", ha="center", va="bottom",
            color="white", fontsize=7
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       color="white", fontsize=7)
    ax.set_ylabel("pass count", color="white", fontsize=11)
    ax.set_title(
        f"{team} - pass cluster distribution\n"
        f"{total} clustered passes · {len(match_ids)} matches",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")
    ax.grid(axis="y", color="white", alpha=0.1, linestyle="--")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()



def plot_flow_centrality(match_ids, team, zip_path,
                         save_path="output/flow_centrality.png",
                         min_passes=5, top_n=20):
    """
    Horizontal bar chart showing flow centrality per player.
    Flow centrality = passes_out + passes_in (weighted by volume).
    Bars are split into passes given (out) vs received (in).
    """
    records  = load_events(match_ids, zip_path)
    team_ev  = team_events(records, team)

    edges = defaultdict(lambda: {"count": 0})

    for _mid, ev in team_ev:
        if ev.get("type", {}).get("id") != 30:
            continue
        p = ev.get("pass", {})
        if p.get("outcome", {}).get("id") in (9, 74, 75, 76, 77):
            continue
        passer    = ev.get("player", {}).get("name")
        recipient = p.get("recipient", {}).get("name") if p.get("recipient") else None
        if not passer or not recipient:
            continue
        edges[(passer, recipient)]["count"] += 1

    passes_out = defaultdict(int)
    passes_in  = defaultdict(int)
    for (passer, recipient), data in edges.items():
        if data["count"] < min_passes:
            continue
        passes_out[passer] += data["count"]
        passes_in[recipient] += data["count"]

    all_players = set(passes_out.keys()) | set(passes_in.keys())
    player_data = []
    for player in all_players:
        out = passes_out.get(player, 0)
        inp = passes_in.get(player, 0)
        player_data.append({
            "player":           player,
            "passes_out":       out,
            "passes_in":        inp,
            "flow_centrality":  out + inp,
        })

    df = pd.DataFrame(player_data).sort_values(
        "flow_centrality", ascending=True
    ).tail(top_n)

    short_names = [n.split()[-1] for n in df["player"]]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    y = np.arange(len(df))

    bars_out = ax.barh(y, df["passes_out"], color="#e53517",
                       alpha=0.85, label="passes given", height=0.6)
    bars_in  = ax.barh(y, df["passes_in"], left=df["passes_out"],
                       color="#43a047", alpha=0.85,
                       label="passes received", height=0.6)

    for i, (out, inp) in enumerate(zip(df["passes_out"], df["passes_in"])):
        total = out + inp
        ax.text(total + 5, i, str(total),
                va="center", color="white", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, color="white", fontsize=10)
    ax.set_xlabel("pass volume", color="white", fontsize=11)
    ax.set_title(
        f"{team} - flow centrality (top {top_n} players)\n"
        f"red = passes given · green = passes received · {len(match_ids)} matches",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")
    ax.grid(axis="x", color="white", alpha=0.1, linestyle="--")
    ax.legend(facecolor="#0d0d1a", labelcolor="white",
              fontsize=10, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    TEAM_MATCHES_CSV = "Olympiacos Piraeus"
    TEAM_JSON = "Olympiacos"
    ZIP_PATH = "data/statsbomb/league_phase.zip"
    MATCHES_CSV = "data/matches.csv"
    OUT = "output/"
    os.makedirs(OUT, exist_ok=True)

    matches_df = pd.read_csv(MATCHES_CSV)
    match_ids = matches_df[
        (matches_df["home"] == TEAM_MATCHES_CSV) | (matches_df["away"] == TEAM_MATCHES_CSV)
    ]["statsbomb"].tolist()
    print(f"{TEAM_MATCHES_CSV}: {len(match_ids)} matches")

    plot_error_heatmap(match_ids, TEAM_JSON, ZIP_PATH,    OUT + "heatmap_errors.png")
    plot_shot_heatmap(match_ids, TEAM_JSON, ZIP_PATH,     OUT + "heatmap_shots.png")
    plot_pressure_heatmap(match_ids, TEAM_JSON, ZIP_PATH, OUT + "heatmap_pressure.png")
    plot_pass_heatmap(match_ids, TEAM_JSON, ZIP_PATH,     OUT + "heatmap_passes.png")
    plot_passing_network(match_ids, TEAM_JSON, ZIP_PATH,  OUT + "passing_network.png")

    plot_pass_clusters(match_ids, TEAM_JSON, ZIP_PATH,
                       OUT + "pass_clusters.png")

    plot_flow_centrality(match_ids, TEAM_JSON, ZIP_PATH,
                         OUT + "flow_centrality.png",
                         min_passes=5, top_n=20)