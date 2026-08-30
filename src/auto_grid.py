"""
Row positions in this broadcast graphic aren't perfectly static across the
video - verified against real footage: from frame 81 onward the row block
sits ~25px lower than in frame 1 (most likely the graphic grows slightly
when a player's row gets highlighted as the active bowler). Column x-bounds
stay put (confirmed via template matching on the header row, dy=0 across the
shift), so only row y-boundaries need re-detecting per frame.

This reuses the same Sobel-edge peak-clustering approach used for the
original manual calibration in calibrate.py, generalized to run
automatically against any frame.
"""
import cv2
import numpy as np


def _cluster(candidates, merge_dist):
    if len(candidates) == 0:
        return []
    clusters = []
    cur = [candidates[0]]
    for v in candidates[1:]:
        if v - cur[-1] <= merge_dist:
            cur.append(v)
        else:
            clusters.append(int(np.mean(cur)))
            cur = [v]
    clusters.append(int(np.mean(cur)))
    return clusters


def detect_row_bounds(frame, x_range, y_search_range, n_rows, merge_dist=6):
    """Detect n_rows+1 horizontal gridline y-positions within y_search_range,
    summing Sobel-Y edge magnitude over x_range. Returns None if it can't find
    exactly the expected count (caller should fall back to the last-known-good
    layout rather than trust a bad detection)."""
    x1, x2 = x_range
    y1, y2 = y_search_range
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_profile = np.sum(np.abs(sobely), axis=1)

    thresh = row_profile.mean() + 2 * row_profile.std()
    candidates = np.where(row_profile > thresh)[0]
    if len(candidates) == 0:
        return None
    clusters = _cluster(list(candidates), merge_dist)

    if len(clusters) != n_rows + 1:
        return None
    return [y1 + c for c in clusters]


def locate_row_bounds(frame, layout, search_pad=120):
    """Try to re-detect row boundaries near where the calibrated layout
    expects them (+/- search_pad), falling back to the calibrated values
    (shifted to align on the actual top-of-grid position) if detection fails."""
    rx, ry, rw, rh = layout["roi"]
    n_rows = len(layout["row_labels"])
    calibrated_abs = [ry + f * rh for f in layout["row_bounds_frac"]]

    col_bounds = layout["col_bounds_frac"]
    x_range = (int(rx + col_bounds[0] * rw), int(rx + col_bounds[-2] * rw))  # span cols 1-10, skip TTL gap
    y_search = (int(calibrated_abs[0] - search_pad), int(calibrated_abs[-1] + search_pad))

    detected = detect_row_bounds(frame, x_range, y_search, n_rows)
    return detected if detected is not None else calibrated_abs
