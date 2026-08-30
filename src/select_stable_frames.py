"""
Step 3: From all extracted frames, pick one sharp frame per "stable segment"
(a run of frames where the scoreboard region isn't changing/animating), so we
don't run OCR on every single frame or on transition/motion-blurred frames.

Usage:
    python src/select_stable_frames.py --frames data/frames --layout config/layout.json
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from grid import load_layout


def roi_crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def select_stable_frames(frames_dir, layout_path, diff_threshold=6.0):
    layout = load_layout(layout_path)
    roi = layout["roi"]

    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {frames_dir}")

    segments = []  # list of lists of (path, gray_roi, blur)
    current_segment = []
    prev_gray = None

    for p in frame_paths:
        img = cv2.imread(str(p))
        crop = roi_crop(img, roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = np.mean(cv2.absdiff(gray, prev_gray))
            if diff > diff_threshold:
                # scoreboard changed or a transition happened - close out segment
                segments.append(current_segment)
                current_segment = []
        current_segment.append((p, gray, blur_score(gray)))
        prev_gray = gray

    if current_segment:
        segments.append(current_segment)

    picks = []
    for seg in segments:
        if not seg:
            continue
        best = max(seg, key=lambda t: t[2])  # sharpest frame in the stable run
        picks.append(best[0])

    print(f"{len(frame_paths)} frames -> {len(segments)} stable segments -> {len(picks)} picked")
    return picks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--layout", default="config/layout.json")
    ap.add_argument("--diff-threshold", type=float, default=6.0,
                     help="Mean pixel diff (0-255) above which the ROI is considered changed")
    args = ap.parse_args()

    picks = select_stable_frames(args.frames, args.layout, args.diff_threshold)
    for p in picks:
        print(p)
