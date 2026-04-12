import argparse
import os
import shutil
import subprocess
from datetime import datetime

import imageio_ffmpeg
import pandas as pd
from utils import MATCHES_CSV, ZIP_PATHS, load_single_match_events

EVENTVIDEO_SCRIPT = "data/wyscout/eventvideo/eventvideo.py"
CONCAT_LIST = "clips.txt"
MERGED_OUTPUT = "merged.mp4"


def timestamp_to_seconds(timestamp):
    """Convert a StatsBomb timestamp like 00:49:05.858 to seconds."""
    parsed = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return (
        parsed.hour * 3600
        + parsed.minute * 60
        + parsed.second
        + parsed.microsecond / 1_000_000
    )


def get_wyscout_match_id(statsbomb_match_id, matches_csv):
    matches_df = pd.read_csv(matches_csv, dtype={"statsbomb": str, "wyscout": str})
    row = matches_df.loc[matches_df["statsbomb"] == str(statsbomb_match_id)]
    if row.empty:
        raise ValueError(f"StatsBomb match_id {statsbomb_match_id} not found in {matches_csv}")
    return row.iloc[0]["wyscout"]


def find_event(events, event_id):
    event_id = str(event_id)

    for ev in events:
        if str(ev.get("id")) == event_id:
            return ev

    if event_id.isdigit():
        event_index = int(event_id)
        for ev in events:
            if ev.get("index") == event_index:
                return ev

    raise ValueError(f"Event {event_id} not found in match data")


def first_half_duration_seconds(events):
    half_end_events = [
        ev
        for ev in events
        if ev.get("type", {}).get("name") == "Half End" and ev.get("period") == 1
    ]
    if not half_end_events:
        raise ValueError("Could not find first-half 'Half End' event")
    return timestamp_to_seconds(half_end_events[0]["timestamp"])


def compute_event_seconds(events, event):
    period = event.get("period")
    event_seconds = timestamp_to_seconds(event["timestamp"])

    if period == 1:
        return event_seconds
    if period == 2:
        return first_half_duration_seconds(events) + event_seconds

    raise ValueError(f"Unsupported event period: {period}")


def merge_clips(clip_paths, concat_list_path, output_path):
    ffmpeg_executable = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    with open(concat_list_path, "w") as f:
        for clip_path in clip_paths:
            f.write(f"file '{os.path.abspath(clip_path)}'\n")

    subprocess.run(
        [
            ffmpeg_executable,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c",
            "copy",
            output_path,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute the video second offset for a StatsBomb event."
    )
    parser.add_argument("match_id", help="StatsBomb match_id")
    parser.add_argument("event_id", help="StatsBomb event id (UUID) or numeric event index")
    parser.add_argument("num_clips", type=int, help="Number of clips to generate")
    args = parser.parse_args()

    wyscout_match_id = get_wyscout_match_id(args.match_id, MATCHES_CSV)

    events = load_single_match_events(args.match_id, ZIP_PATHS)
    event = find_event(events, args.event_id)
    event_seconds = compute_event_seconds(events, event)

    clip_paths = []
    for i in range(args.num_clips):
        second = int(event_seconds) + i * 10
        clip_path = f"clip{i}.mp4"
        clip_paths.append(clip_path)
        subprocess.run(
            [
                "python3",
                EVENTVIDEO_SCRIPT,
                str(wyscout_match_id),
                str(second),
                clip_path,
            ],
            check=True,
        )

    merge_clips(clip_paths, CONCAT_LIST, MERGED_OUTPUT)


if __name__ == "__main__":
    main()
