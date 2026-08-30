"""
Step 2: Interactive calibration - run this once (per broadcast layout) against a
representative frame to define the scoreboard grid geometry. Saves config/layout.json.

Must be run locally in a terminal with a display (it opens OpenCV windows) -
this is not something that can run headlessly.

Usage:
    python src/calibrate.py --frame data/frames/frame_000001.png
"""
import argparse
import json
from pathlib import Path

import cv2

# Row layout observed in the sample scoreboard: three players (symbol row +
# running-total row each) plus a team total row (symbol row + total row).
# Edit this list if a different broadcast layout has a different row structure.
DEFAULT_ROW_LABELS = [
    ("J", "symbols"), ("J", "score"),
    ("V", "symbols"), ("V", "score"),
    ("P", "symbols"), ("P", "score"),
    ("T", "symbols"), ("T", "score"),
]
DEFAULT_COL_LABELS = [str(i) for i in range(1, 11)] + ["TTL"]


def click_points(window_name, img, n_points, instruction):
    points = []
    disp = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append(x)
            cv2.line(disp, (x, 0), (x, disp.shape[0]), (0, 255, 0), 1)
            cv2.imshow(window_name, disp)

    print(f"{instruction} ({n_points} clicks needed)")
    cv2.imshow(window_name, disp)
    cv2.setMouseCallback(window_name, on_click)
    while len(points) < n_points:
        if cv2.waitKey(20) == 27:  # Esc aborts
            raise SystemExit("Calibration aborted")
    return sorted(points)


def click_points_y(window_name, img, n_points, instruction):
    points = []
    disp = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append(y)
            cv2.line(disp, (0, y), (disp.shape[1], y), (0, 255, 0), 1)
            cv2.imshow(window_name, disp)

    print(f"{instruction} ({n_points} clicks needed)")
    cv2.imshow(window_name, disp)
    cv2.setMouseCallback(window_name, on_click)
    while len(points) < n_points:
        if cv2.waitKey(20) == 27:
            raise SystemExit("Calibration aborted")
    return sorted(points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="Path to a representative frame image")
    ap.add_argument("--out", default="config/layout.json")
    args = ap.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise SystemExit(f"Could not read image: {args.frame}")

    print("Drag a box around the WHOLE scoreboard grid (columns 1-10 + TTL, "
          "all player rows) - do not include the lane number / handicap boxes. "
          "Press ENTER/SPACE to confirm, 'c' to cancel.")
    roi = cv2.selectROI("Select scoreboard grid ROI", img, showCrosshair=True)
    cv2.destroyWindow("Select scoreboard grid ROI")
    x, y, w, h = roi
    if w == 0 or h == 0:
        raise SystemExit("Empty ROI selected, aborting")

    crop = img[y:y + h, x:x + w]
    scale = 2 if max(crop.shape[:2]) < 1200 else 1
    crop_disp = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale))

    n_cols = len(DEFAULT_COL_LABELS)
    n_rows = len(DEFAULT_ROW_LABELS)

    col_xs = click_points(
        "Click column boundaries",
        crop_disp,
        n_cols + 1,
        f"Click the {n_cols + 1} vertical column boundary lines, LEFT to RIGHT "
        f"(left edge of col 1, ..., right edge of {DEFAULT_COL_LABELS[-1]})",
    )
    cv2.destroyWindow("Click column boundaries")

    row_ys = click_points_y(
        "Click row boundaries",
        crop_disp,
        n_rows + 1,
        f"Click the {n_rows + 1} horizontal row boundary lines, TOP to BOTTOM "
        f"(top of row {DEFAULT_ROW_LABELS[0]}, ..., bottom of row {DEFAULT_ROW_LABELS[-1]})",
    )
    cv2.destroyWindow("Click row boundaries")
    cv2.destroyAllWindows()

    # normalize clicked pixel coords (in the possibly-upscaled display) back to
    # fractions of the ROI, so the grid can be applied to ROIs of any size later.
    col_fracs = [px / (w * scale) for px in col_xs]
    row_fracs = [py / (h * scale) for py in row_ys]

    print("Now drag a box around the LANE NUMBER (top-left '6' box). "
          "Press ENTER/SPACE to confirm.")
    lane_roi = cv2.selectROI("Select lane number ROI", img, showCrosshair=True)
    cv2.destroyWindow("Select lane number ROI")

    print("Now drag a box around the PLAYER/TEAM NAME ('TARUN'). "
          "Press ENTER/SPACE to confirm.")
    name_roi = cv2.selectROI("Select name ROI", img, showCrosshair=True)
    cv2.destroyWindow("Select name ROI")
    cv2.destroyAllWindows()

    layout = {
        "roi": [int(x), int(y), int(w), int(h)],
        "lane_roi": [int(v) for v in lane_roi],
        "name_roi": [int(v) for v in name_roi],
        "source_frame": str(args.frame),
        "col_bounds_frac": col_fracs,
        "row_bounds_frac": row_fracs,
        "col_labels": DEFAULT_COL_LABELS,
        "row_labels": [{"player": p, "type": t} for p, t in DEFAULT_ROW_LABELS],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(layout, indent=2))
    print(f"Saved layout to {out_path}")


if __name__ == "__main__":
    main()
