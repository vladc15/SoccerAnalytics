# SoccerAnalytics

## `get_video.py`

`get_video.py` generates one or more Wyscout video clips for a StatsBomb event and
then merges them into a single longer clip.

### Dependency

To stitch the generated clips into `merged.mp4`, install:

```bash
python -m pip install imageio-ffmpeg
```

### Usage

```bash
python get_video.py <statsbomb_match_id> <event_id> <N>
```

Arguments:

- `statsbomb_match_id`: the StatsBomb match id, used to find the corresponding
  Wyscout match id in `data/matches.csv`
- `event_id`: the StatsBomb event UUID, or the numeric StatsBomb event `index`
- `N`: the number of clips to generate, each of them is 10s long

### How it works

1. Looks up the equivalent `wyscout` match id in `data/matches.csv`
2. Loads the StatsBomb match from:
   - `data/statsbomb/league_phase.zip`
   - `data/statsbomb/playoffs.zip`
3. Finds the requested event
4. Converts the event timestamp to seconds:
   - if the event is in period 1, it uses the event timestamp directly
   - if the event is in period 2, it adds the first-half `Half End` timestamp to
     the event timestamp
5. Floors the computed second to an integer
6. Calls:

```bash
python3 data/wyscout/eventvideo/eventvideo.py {wyscout_match_id} {second} clip{i}.mp4
```

for each `i` from `0` to `N-1`, increasing the second by `10` each time
7. Merges the generated clips into `merged.mp4`

### Outputs

- individual clips: `clip0.mp4`, `clip1.mp4`, ..., `clip{N-1}.mp4`
- concat manifest: `clips.txt`
- merged video: `merged.mp4`

### Example

```bash
python3 get_video.py 4028972 5cc9c8a3-f82f-4b70-827d-462f1e4df346 8 
```

This generates:

- `clip0.mp4` at the computed event second
- `clip1.mp4` starting `10` seconds later
- `clip2.mp4` starting `20` seconds later
...
- `clip7.mp4` starting `70` seconds later
- `merged.mp4` containing all three clips concatenated
