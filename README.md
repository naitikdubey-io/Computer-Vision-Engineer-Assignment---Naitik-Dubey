# 🎳 Bowling Scoreboard Extraction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.9%2B-5C3EE8)
![Status](https://img.shields.io/badge/status-validated-brightgreen)

Pulls structured data — per-frame rolls, running totals, player names, lane
number — out of broadcast video of a bowling scoreboard, using a custom
glyph classifier instead of general OCR (which misreads this broadcast's
font — see [Approach](#approach)).

<img src="docs/screenshots/03_detected_grid.png" width="720" alt="Calibrated grid detected on a real input frame">

*The calibrated cell grid detected on a real frame from the source video.*

## Quick start

**1. Get the code**

```bash
git clone https://github.com/naitikdubey-io/Computer-Vision-Engineer-Assignment---Naitik-Dubey.git
cd Computer-Vision-Engineer-Assignment---Naitik-Dubey
```

**2. Install ffmpeg and Tesseract** — two external tools the pipeline shells
out to, not Python packages, so `pip` alone won't get them:

```bash
winget install Gyan.FFmpeg UB-Mannheim.TesseractOCR
```
*(macOS: `brew install ffmpeg tesseract` — Linux: `apt install ffmpeg tesseract-ocr`)*

**3. Open a new terminal.** This matters: a terminal already open when you
installed step 2 won't see the update. Do everything below in a fresh one.

**4. Install the Python dependencies**

```bash
pip install -r requirements.txt
```

**5. Add the video.** Place it at `data/bowling_scoreboard.mp4` — exactly
that name, in the `data` folder.

**6. Extract frames from the video** — samples it down to a few frames per
second, since a scoreboard graphic doesn't change every frame:

```bash
cd src
python extract_frames.py --video ../data/bowling_scoreboard.mp4 --out ../data/frames --fps 3
```

**7. Run the pipeline** — reads those frames and writes the extracted data:

```bash
python pipeline.py --frames ../data/frames --out ../output/scorecards.json
```

That's it — check `output/scorecards.json` for the result. Hitting an
error? [Troubleshooting](#user-content-troubleshooting) below covers the
things most likely to trip up a first run.

## Output

[`output/scorecards.json`](output/scorecards.json) is committed as-is from
a real run, so the result is visible without setting anything up — running
the pipeline yourself regenerates that same file and should match it
exactly.

One JSON entry per distinct scorecard state found in the video:

```json
{
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

`validation_issues` flags anything a bowling-scoring cross-check found
suspect — a non-empty entry means "look at this one," not necessarily wrong.

## Approach

General OCR (Tesseract) was tried first and rejected — it misread this
broadcast's font consistently, reading a clean `9` as nothing at all in
every mode tried. Since the vocabulary is tiny (digits, `X`, `/`, `-`) and
the font is fixed, a small **template-matching classifier** trained on
glyphs from this video's own footage turned out simpler and far more
accurate.

The rest of the pipeline exists because the footage kept breaking
assumptions that seemed safe: the grid isn't pixel-static (the active
bowler's row grows and highlights), a transient animation intermittently
covers part of the grid, and the broadcast shows *provisional* totals
mid-frame.

## Architecture

Nine stages, each one earned by a real failure mode found in the footage:

```
video
  │
  ├─ extract_frames.py        sample to frames (ffmpeg)
  ├─ select_stable_frames.py  dedupe near-identical frames, keep sharpest per segment
  ├─ layout_selector.py       skip non-scoreboard frames (bumpers/cartoons);
  │                           pick the calibrated layout for whichever row
  │                           is currently highlighted
  ├─ overlay.py               mask cells covered by the transient pin-fall
  │                           animation graphic
  ├─ grid.py                  crop each cell using the calibrated layout
  ├─ recognize.py             preprocess each crop (contrast, binarize,
  │                           cheap blank check)
  ├─ glyph_classifier.py      recognize each cell — custom template
  │                           matching, with logic to split touching digits
  ├─ bowling_rules.py         cross-check the result against scoring rules
  └─ pipeline.py              orchestrates all of the above
                                        │
                                        ▼
                              output/scorecards.json
```

## Project layout

```
src/       pipeline code (see Architecture above for what each module does)
config/    calibrated grid geometry + glyph template bank
data/      source video + extracted frames (not committed)
output/    scorecards.json — the result
docs/      documentation PDF, screenshots
```

## Docs

- **[docs/documentation.pdf](docs/documentation.pdf)** — screenshots + explanations (input, code running, detection, output)

---

<details>
<summary id="troubleshooting"><b>Troubleshooting</b></summary>

**`ModuleNotFoundError` despite installing requirements** — you likely have
more than one Python on this machine (conda + system Python, a venv, etc.)
and `pip install` went to a different one than `python` is running. Check
with `python -c "import sys; print(sys.executable)"` before and after
installing. If your prompt shows `(base)`, either `pip install -r
requirements.txt` inside that conda env, or `conda deactivate` and use your
system Python.

**ffmpeg/tesseract "not found"** — [`src/tool_paths.py`](src/tool_paths.py)
checks PATH plus a few common install locations automatically, so this
usually means one of them genuinely isn't installed yet (see Quick start
above), or a fresh terminal hasn't been opened since installing (Windows
shells cache PATH until restarted). If it's installed somewhere unusual,
add that path to `_FFMPEG_PATTERNS` / `_TESSERACT_PATTERNS` in that file.

**Video file not found** — it needs to be at `data/bowling_scoreboard.mp4`
exactly. Browser downloads often append `(1)` if a file with that name
already exists in Downloads — rename it, or a name with spaces/parentheses
will need quoting in every command.

**Pipeline seems to hang** — it prints nothing until finished. On this
video that's ~5-10s for frame extraction, ~30-60s for the full pipeline.
Let it finish rather than interrupting.

</details>
