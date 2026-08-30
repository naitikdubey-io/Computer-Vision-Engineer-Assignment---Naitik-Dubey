# Bowling Scoreboard Extraction

Extracts structured scorecard data — per-frame rolls, running totals, player
names, and lane number — from broadcast video of a bowling scoreboard.

Built for one specific broadcast graphic rather than bowling scoreboards in
general: the pipeline is calibrated against this graphic's exact grid
geometry and font, and recognizes characters with a small template-matching
classifier trained on that font, rather than general-purpose OCR (which
misreads it — see [Approach](#approach) below).

A full build log — the approach, every problem hit along the way, what
changed and why, and final validation results — is in
[`docs/BUILD_LOG.md`](docs/BUILD_LOG.md).

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) on PATH (frame extraction)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) on PATH
  (used only for the free-text player-name field; all score/symbol/lane
  recognition uses the custom classifier)

```bash
pip install -r requirements.txt
```

If `ffmpeg`/`tesseract` are installed but not resolving on PATH (a common
issue right after installing on Windows — the current shell's PATH doesn't
refresh until it's restarted), edit the fallback paths in
[`src/tool_paths.py`](src/tool_paths.py) to point at your install locations;
the scripts locate the binaries through that module rather than assuming
PATH is current.

## Usage

The source video isn't committed to this repo (see `.gitignore`) — place it
at `data/bowling_scoreboard.mp4`, or pass `--video` with any path.

```bash
cd src

# 1. Sample the video into frames (3fps is enough for a scoreboard graphic)
python extract_frames.py --video ../data/bowling_scoreboard.mp4 --out ../data/frames --fps 3

# 2. Run the full pipeline
python pipeline.py --frames ../data/frames --out ../output/scorecards.json
```

Output is one JSON entry per distinct scorecard state found in the video —
not one per frame, since the pipeline first collapses long runs of
visually-identical frames down to a single representative one (see
[Architecture](#architecture)). Each entry looks like:

```json
{
  "frame_file": "..\\data\\frames\\frame_000001.png",
  "state": "none",
  "lane": "6",
  "name": "TARUN",
  "players": {
    "J": {
      "symbols": ["X", "5-", "-7", "4-", "", "", "", "", "", ""],
      "scores":  [15, 20, 27, 31, null, null, null, null, null, null],
      "ttl": 31
    }
  },
  "validation_issues": {}
}
```

`validation_issues` flags anything a cross-check against bowling scoring
rules found suspect (a non-monotonic running total, a TTL that doesn't
match the latest frame or a known provisional-total pattern) — a non-empty
entry means "look at this one," not necessarily "this is wrong."

### Recalibrating for a different broadcast graphic

The grid geometry, highlight-state layouts, and glyph templates are all
specific to this video's graphic. To point the pipeline at a different
broadcast:

1. `python calibrate.py --frame <a representative frame>` — click-through
   tool that measures the grid's column/row boundaries and saves
   `config/layout.json`.
2. `python build_templates.py` — bootstraps the glyph template bank from a
   frame with known ground truth (edit the ground-truth table at the top of
   the script to match your frame).
3. If the graphic also shifts layout when a row is highlighted (as this one
   does), calibrate each additional state the same way as step 1, saving to
   `config/layout_<STATE>.json`, and register it in
   [`src/layout_selector.py`](src/layout_selector.py).

## Architecture

```
video
  -> extract_frames.py       sample to frames (ffmpeg)
  -> select_stable_frames.py dedupe near-identical frames, keep sharpest per segment
  -> layout_selector.py      skip non-scoreboard frames (bumpers/cartoons);
                              pick the right calibrated layout for whichever
                              row is currently highlighted
  -> overlay.py               mask cells covered by the transient pin-fall
                              animation graphic
  -> grid.py                  crop each cell using the calibrated layout
  -> recognize.py             preprocess each crop (contrast, binarize,
                              cheap blank check)
  -> glyph_classifier.py      recognize each cell — custom template matching,
                              with logic to split touching/merged digits
  -> bowling_rules.py         cross-check the result against scoring rules
  -> output/scorecards.json
```

## Approach

General OCR (Tesseract) was tried first and rejected: it misread this
broadcast's stylized font consistently — a clean, unambiguous `9` read as
nothing at all, in every page-segmentation mode tried. Since the actual
character vocabulary on the board is tiny (digits, `X`, `/`, `-`) and the
font is fixed (rendered by broadcast graphics software, not photographed),
a small template-matching classifier trained on glyphs pulled from this
video's own footage turned out to be both simpler and dramatically more
accurate.

The rest of the pipeline exists because the footage kept violating
assumptions that seemed safe at first: the grid isn't pixel-static (the
active bowler's row grows and highlights, shifting every row below it), a
transient animation intermittently covers part of the grid at an unpredictable
position, and the broadcast graphic shows *provisional* totals mid-frame
that look like errors until you know the pattern. The full account of each
of these, and how they were diagnosed and fixed, is in the build log.

## Project structure

```
src/
  extract_frames.py        ffmpeg wrapper - samples the video into frames
  select_stable_frames.py  dedupes near-identical frames
  layout_selector.py       scoreboard gate + highlight-state detection
  overlay.py                detects/masks the pin-fall animation
  grid.py                    turns a calibrated layout into per-cell pixel boxes
  recognize.py               preprocessing: contrast, binarization, blank check
  glyph_classifier.py        the custom recognizer + digit-split logic
  bowling_rules.py           cross-validation against scoring rules
  pipeline.py                 orchestrates all of the above
  calibrate.py, auto_grid.py, build_templates.py, tool_paths.py
                              calibration & setup utilities

config/
  layout.json, layout_J.json, layout_V.json   calibrated grid geometry per state
  templates/                                   glyph template bank

data/    source video + extracted frames (not committed - see .gitignore)
output/  scorecards.json - the final structured result
docs/    build log / documentation
```
