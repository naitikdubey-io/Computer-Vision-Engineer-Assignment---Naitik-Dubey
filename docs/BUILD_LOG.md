# Build Log — Bowling Scoreboard Extraction

A chronological account of the approach, every problem hit, what changed
and why, and the final validated result.

## The task

Build a CV system that watches a bowling broadcast and pulls the
scoreboard's data — per-frame rolls, running totals, player names, lane
number — into structured, checkable output.

The scoreboard is a fixed on-screen graphic (a "TV bug"): a grid of ten
frames per player, three players plus a team-total row, a lane number, and
whichever player's name is currently active. It looks static. Nothing about
it turned out to be static, which is most of what this log is about.

## Approach

**General OCR was tried first and rejected.** Tesseract, even with a
restricted digit whitelist and every page-segmentation mode tried, misread
this broadcast's font consistently — a clean, high-contrast `9` was read as
nothing at all, in every mode. Since the actual character vocabulary is
tiny (digits, `X`, `/`, `-`) and the font is fixed (rendered by broadcast
software, not photographed), a small template-matching classifier was built
from scratch instead: segment each cell into connected components,
normalize to a canonical size, match against templates pulled from this
video's own font. This was both simpler and far more reliable than tuning
Tesseract further.

## Pipeline architecture

```
video
  -> extract_frames.py        sample to frames (ffmpeg, 3fps -> 174 frames)
  -> select_stable_frames.py  dedupe near-identical frames, keep sharpest per segment
  -> layout_selector.py       scoreboard gate (skip bumpers/cartoons cut into
                               the broadcast); pick the calibrated layout for
                               whichever row is currently highlighted
  -> overlay.py                mask cells covered by the transient pin-fall
                               animation
  -> grid.py                   crop each cell using calibrated boundaries
  -> recognize.py              preprocess: contrast, binarize, blank check
  -> glyph_classifier.py       recognize each cell (custom template matching)
  -> bowling_rules.py          cross-check against bowling scoring rules
  -> output/scorecards.json
```

## Problems faced, and what changed

In the order they were actually discovered.

### 1. General OCR fails on this broadcast's font

Tesseract misread specific glyphs consistently — most tellingly, a clean
`9` read as nothing at all, in every PSM mode.

**Change in approach:** dropped general OCR for digit/symbol cells. Built a
template-matching glyph classifier: segment each cell into connected
components, normalize, match against templates pulled from this video's own
font.

### 2. Segmentation noise from cell borders

Connected-component segmentation initially picked up grid-line bleed and
cell-border artifacts as if they were characters.

**Fix:** layered filters tuned against real failure cases — thin slivers
(grid lines), hollow full-cell outlines (the team-total row's colored block
boundary), and sparse low-fill blobs (corner bleed) are discarded, but only
when they touch the crop border, since a real glyph can legitimately reach
an edge too (a `/`'s diagonal stroke, for instance).

### 3. Adjacent score digits render touching, not separate

Two-digit totals like `20` or `48` merge into one connected component in
this font at this size.

- **Iteration 1 — ink-valley splitting:** find the sparsest vertical column,
  cut there. Worked for evenly-sized pairs, failed on asymmetric ones like
  `31` — `1` is much narrower than `3`, so the true valley sits far
  off-center, and `3`'s own internal curve can dip lower in ink than the
  actual character boundary.
- **Iteration 2 — classification-guided splitting:** try many candidate cut
  points, classify both halves, keep the best. Ranking by the *minimum* of
  the two confidences initially picked a wrong split (`2`+`3` at
  0.78/0.79) over the correct one (`2`+`8` at 0.93/0.77) — switched to
  ranking by the *sum*, which correctly rewards one very confident half
  over two merely-decent ones.
- **Iteration 3 — width override:** even with sum-ranking, a
  confidently-wrong single-character read (a `27` blob reading as a lone
  `2` at 0.89) could still beat a correct split's weaker half. Added a hard
  override: a blob measurably too wide for one digit (single digits
  measured 143–172px; genuine merges measured 300px+) accepts the split
  outright, bypassing the confidence contest.

### 4. The scoreboard grid isn't static

Assumed one fixed grid layout. Real footage showed the active bowler's row
grows taller and gets highlighted, pushing every row below it down by a
consistent offset — and the team-total row's own styling changes depending
on whether anyone is highlighted at all.

**Fix:** measured the row-shift pattern from one calibrated highlighted
frame, confirmed it generalized (predicted values matched a second
highlighted state to the pixel), and calibrated three discrete layouts: no
highlight, "J" highlighted, "V" highlighted. Scanned every frame in the
video to confirm the third player never actually gets highlighted in this
clip, so a fourth layout wasn't needed.

### 5. A transient animation obscures part of the grid

A "which pins fell" graphic pops in after most rolls. First assumed fixed
screen position; a second sighting showed it tracking wherever the action
just happened (sometimes over the top rows, sometimes the bottom).

**Fix:** replaced the fixed-box idea with a per-row-block color check (each
block's known clean background color vs. its current reading), so
detection works regardless of where the overlay sits. Obscured cells are
left blank rather than fed to the classifier.

### 6. Lane number and player name fields

The lane digit uses a more cursive style than the score/symbol fonts.
Separately, the free-text name field kept getting a stray trailing
character from Tesseract.

**Fix:** one small dedicated template for the lane digit's style. Traced
the name-field artifact to a contrast-enhancement step manufacturing a fake
edge out of the mostly-blank background — fixed by filtering candidate text
regions by height before cropping.

### 7. The running-total (TTL) column

Initially wildly wrong (`31` read as `831`). Once fixed, some TTL reads
still failed validation — but those weren't OCR errors.

- **Fix 1:** the TTL cell's crop included a ~80px gap before the actual
  number box; that gap binarized into a false leading digit. Corrected the
  column boundary.
- **Fix 2 (a false alarm, not a bug):** bowling broadcasts show a
  *provisional* total — current total plus an in-progress roll's value —
  before a frame is fully scored (a strike shows total+10 immediately).
  The validator didn't know this and flagged correct reads as mismatches.
  Taught it to recognize both known provisional forms (a pending strike, a
  single partial roll).

### 8. Blank cells occasionally misread as a dash

The contrast-enhancement step could manufacture a fake edge out of a blank
cell's background gradient alone, occasionally surviving noise filters and
misclassifying as `-`.

**Fix:** added a cheaper, more reliable blank check on raw pixel variance,
run before the enhancement step — with separate thresholds for the score
row and symbol row, since their font sizes give genuinely different
variance baselines.

### 9. Known limitation — one genuine video-compression artifact

One TTL cell, across several consecutive frames in the same stable
segment, shows two different digit states visibly superimposed in the same
pixels — confirmed by direct visual inspection.

This is inter-frame video compression baking two different broadcast states
into one decoded frame; the source pixels genuinely contain two overlapping
characters. No classifier can resolve that into one correct digit. The
validator correctly flags it as suspect rather than guessing — that's the
intended behavior, not a failure of it.

## Final validation

- **34/34** cells correct against manually-verified ground truth on frame 1
  (every symbol, score, and TTL cell).
- **174 → 121** frames extracted (3fps) → frames that pass the scoreboard
  gate (the rest are cut-in bumpers/cartoons).
- **6** distinct scorecard states found across the full 58s video.
- **5/6** states with zero validation warnings — the 6th is the confirmed
  compression artifact above.

**Cross-checked at 10fps** (578 frames vs. 174, 3x finer sampling) to
confirm nothing was missed between samples: found 8 states instead of 6,
but all 8 carry identical data to the original 6 — no new rolls, no new
values, just finer-grained duplicates of transitions already captured. The
same compression artifact reproduced in the same segment, further
confirming it's a property of the source video rather than a flaky read.

| Stage | 3fps run | 10fps run |
|---|---|---|
| Frames extracted | 174 | 578 |
| Stable segments | 26 | 51 |
| Distinct scorecard states | 6 | 8 |
| Game states missed | — | **0** |
