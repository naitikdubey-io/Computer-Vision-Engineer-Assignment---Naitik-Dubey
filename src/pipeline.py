"""
End-to-end: for each selected stable frame, skip it if it isn't actually
showing the scoreboard (the source recording cuts to bumpers/cartoons
between scoreboard segments), pick the right calibrated layout for whichever
highlight state it's in, mask out any cells the transient pin-fall overlay is
covering, crop and recognize every cell with the glyph classifier, assemble a
structured scorecard snapshot, and validate it.

Usage:
    python src/pipeline.py --frames data/frames --out output/scorecards.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from grid import cells_for_layout, crop_cell
from recognize import preprocess, preprocess_variants, is_likely_blank
from glyph_classifier import GlyphClassifier
from layout_selector import LayoutSelector
from bowling_rules import validate_totals, validate_ttl
import overlay as overlay_mod
from select_stable_frames import select_stable_frames


def ocr_frame(frame, layout, state, classifier):
    """Returns {player: {"symbols": [10 str], "scores": [10 Optional[int]], "ttl": Optional[int]}}"""
    by_player = defaultdict(lambda: {"symbols": [None] * 10, "scores": [None] * 10, "ttl": None})
    obscured = overlay_mod.obscured_blocks(frame, layout, state)

    for cell in cells_for_layout(layout):
        player = cell.row_label["player"]
        kind = cell.row_label["type"]

        if overlay_mod.cell_is_obscured(cell, obscured):
            continue  # leave as None/default rather than OCR-ing pin-graphic pixels

        crop = crop_cell(frame, cell)
        if is_likely_blank(crop, kind):
            text = ""
        else:
            variants = preprocess_variants(crop)
            text = classifier.recognize_cell_best_of(variants, kind)

        if cell.col_label == "TTL":
            if kind == "score" and text.isdigit():
                by_player[player]["ttl"] = int(text)
            continue

        col_idx = int(cell.col_label) - 1
        if kind == "symbols":
            by_player[player]["symbols"][col_idx] = text
        else:
            by_player[player]["scores"][col_idx] = int(text) if text.isdigit() else None

    return dict(by_player)


def _tight_text_crop(crop, min_height_frac=0.35, pad=5):
    """Crop tightly to the actual text before handing to Tesseract. A plain
    bbox-of-foreground crop doesn't work here: Otsu on the raw gradient
    background misclassifies much of it as foreground, and even after the
    CLAHE-based preprocess() there's a thin full-width noise streak (a
    CLAHE-amplified background seam) that keeps the naive bbox pinned to the
    full crop width - verified on frame_000001's name field, where it also
    produced a stray trailing character in the OCR output. Filtering to only
    components at least min_height_frac of the crop's height drops that
    streak (and the background noise) while keeping actual letters, which are
    much taller than either artifact."""
    bin_ = preprocess(crop, upscale=1)
    inv = cv2.bitwise_not(bin_)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = bin_.shape
    boxes = [stats[i] for i in range(1, n) if stats[i][3] >= min_height_frac * h]
    if not boxes:
        return crop
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return crop[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def read_lane_and_name(frame, layout, classifier):
    """Lane number uses the same stylized broadcast font as the rest of the
    board (plain Tesseract misreads it, same as it did for cell digits before
    the glyph classifier), so it's recognized the same way. The player name
    is free text (outside the tiny digit/symbol vocabulary the classifier was
    built for), so Tesseract is still used there, just on a tightly-cropped
    region to avoid the background-noise artifacts described in
    _tight_text_crop."""
    lane_text = name_text = None
    if "lane_roi" in layout:
        x, y, w, h = layout["lane_roi"]
        variants = preprocess_variants(frame[y:y + h, x:x + w])
        lane_text = classifier.recognize_lane_best_of(variants)
    if "name_roi" in layout:
        x, y, w, h = layout["name_roi"]
        tight = _tight_text_crop(frame[y:y + h, x:x + w])
        gray = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
        name_text = pytesseract.image_to_string(gray, config="--psm 7").strip()
    return lane_text, name_text


def run(frames_dir, out_path, diff_threshold, config_dir="../config", calibration_frame="../data/frames/frame_000001.png"):
    selector = LayoutSelector(config_dir=config_dir, calibration_frame=calibration_frame)
    classifier = GlyphClassifier(template_dir=f"{config_dir}/templates")

    # stability filtering uses the "no highlight" layout's ROI as a reference
    # region - good enough for detecting motion/scene-changes even though it
    # doesn't perfectly bound every highlight state
    picks = select_stable_frames(frames_dir, f"{config_dir}/layout.json", diff_threshold)

    results = []
    skipped_non_scoreboard = 0
    for frame_path in picks:
        frame = cv2.imread(str(frame_path))
        layout, state = selector.select(frame)
        if layout is None:
            skipped_non_scoreboard += 1
            continue

        lane, name = read_lane_and_name(frame, layout, classifier)
        players = ocr_frame(frame, layout, state, classifier)

        issues = {}
        for player, data in players.items():
            probs = validate_totals(data["scores"]) + validate_ttl(data["scores"], data["ttl"], data["symbols"])
            if probs:
                issues[player] = probs

        results.append({
            "frame_file": str(frame_path),
            "state": state,
            "lane": lane,
            "name": name,
            "players": players,
            "validation_issues": issues,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} scorecard snapshots to {out_path} "
          f"({skipped_non_scoreboard} non-scoreboard frames skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="../data/frames")
    ap.add_argument("--out", default="../output/scorecards.json")
    ap.add_argument("--config-dir", default="../config")
    ap.add_argument("--calibration-frame", default="../data/frames/frame_000001.png")
    ap.add_argument("--diff-threshold", type=float, default=6.0)
    args = ap.parse_args()
    run(args.frames, args.out, args.diff_threshold, args.config_dir, args.calibration_frame)
