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

import under_pressure_stats as ups
from utils import (
    MATCHES_CSV,
    TEAM_MATCHES_CSV,
    ZIP_PATHS,
    draw_pitch,
    load_events,
    make_heatmap,
    team_events,
)

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

def conceded_pressure_mistake_locations(match_ids, team, zip_path):
    """
    Return locations of pressured mistakes that led to a goal or penalty
    conceded before the team regained possession.
    """
    coords = []
    for item in conceded_pressure_mistake_events(match_ids, team, zip_path):
        loc = item["event"].get("location")
        if loc:
            coords.append((loc[0], loc[1]))

    return coords


CMAP_RED   = LinearSegmentedColormap.from_list("red_hm",   ["#1a1a2e", "#6a0f2a", "#c0392b", "#e74c3c", "#ff8a80"])
CMAP_BLUE  = LinearSegmentedColormap.from_list("blue_hm",  ["#1a1a2e", "#0d2b5e", "#1565c0", "#42a5f5", "#b3e5fc"])
CMAP_GREEN = LinearSegmentedColormap.from_list("green_hm", ["#1a1a2e", "#0a3d1f", "#1b5e20", "#43a047", "#b9f6ca"])
CMAP_GOLD  = LinearSegmentedColormap.from_list("gold_hm",  ["#1a1a2e", "#4a2c00", "#bf8000", "#ffca28", "#fff9c4"])


def plot_error_heatmap(match_ids, team, zip_path, save_path="heatmap_errors.png"):
    records = load_events(match_ids, zip_path)
    coords = extract_locations(records, team, type_ids={38, 3})
    conceded_coords = conceded_pressure_mistake_locations(match_ids, team, zip_path)

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

    if conceded_coords:
        cxs, cys = zip(*conceded_coords)
        ax.scatter(
            cxs,
            cys,
            s=120,
            color="#ffca28",
            alpha=0.95,
            edgecolors="#7f0000",
            linewidths=1.0,
            marker="X",
            zorder=5,
        )

    ax.set_title(
        f"{team} - errors on field (miscontrol + dispossessed)\n"
        f"n={len(coords)} events · {len(conceded_coords)} pressured mistakes led to goal/penalty conceded · {len(match_ids)} matches",
        color="white", fontsize=13, pad=12, fontweight="bold"
    )

    legend_elements = [
        mpatches.Patch(facecolor="white", alpha=0.35, label=f"Errors ({len(coords)})"),
        plt.Line2D(
            [0], [0],
            marker="X",
            color="none",
            markerfacecolor="#ffca28",
            markeredgecolor="#7f0000",
            markersize=10,
            label=f"Pressure mistake -> goal/penalty conceded ({len(conceded_coords)})",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=10,
        framealpha=0.8,
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


def plot_pressure_success_scatter(zip_paths, save_path="output/pressure_success_scatter.png", min_episodes=50):
    """
    Competition-wide scatter plot of under-pressure volume vs success rate.

    X axis: pressured episodes
    Y axis: success_under_pressure_pct
    Only includes players with at least `min_episodes` pressured episodes.
    Olympiacos players are highlighted and labeled.
    """
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
                zorder=5,
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
        f"x = volume, y = success rate, min {min_episodes} episodes",
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


# Under-pressure plotting now lives in `under_pressure_stats.py`.
conceded_pressure_mistake_events = ups.conceded_pressure_mistake_events
conceded_pressure_mistake_locations = ups.conceded_pressure_mistake_locations
plot_pressure_heatmap = ups.plot_pressure_heatmap
plot_pressure_success_scatter = ups.plot_pressure_success_scatter


if __name__ == "__main__":
    TEAM_JSON = "Olympiacos"
    OUT = "output/"
    os.makedirs(OUT, exist_ok=True)

    matches_df = pd.read_csv(MATCHES_CSV)
    raw_match_ids = matches_df[
        (matches_df["home"] == TEAM_MATCHES_CSV) | (matches_df["away"] == TEAM_MATCHES_CSV)
    ]["statsbomb"].astype(str).tolist()
    available = set()
    for one_zip_path in ZIP_PATHS:
        with zipfile.ZipFile(one_zip_path) as zf:
            available.update(zf.namelist())
    match_ids = [mid for mid in raw_match_ids if f"{mid}.json" in available]
    print(f"{TEAM_MATCHES_CSV}: {len(match_ids)} matches")

    plot_error_heatmap(match_ids, TEAM_JSON, ZIP_PATHS,    OUT + "heatmap_errors.png")
    plot_shot_heatmap(match_ids, TEAM_JSON, ZIP_PATHS,     OUT + "heatmap_shots.png")
    plot_pressure_heatmap(match_ids, TEAM_JSON, ZIP_PATHS, OUT + "heatmap_pressure.png")
    plot_pass_heatmap(match_ids, TEAM_JSON, ZIP_PATHS,     OUT + "heatmap_passes.png")
    plot_passing_network(match_ids, TEAM_JSON, ZIP_PATHS,  OUT + "passing_network.png")

    plot_pass_clusters(match_ids, TEAM_JSON, ZIP_PATHS,
                       OUT + "pass_clusters.png")

    plot_flow_centrality(match_ids, TEAM_JSON, ZIP_PATHS,
                         OUT + "flow_centrality.png",
                         min_passes=5, top_n=20)

    plot_pressure_success_scatter(
        ZIP_PATHS,
        OUT + "pressure_success_scatter.png",
        min_episodes=50,
    )
