"""
Demo-only helper: pops up windows showing the calibrated grid overlaid on a
frame, and the overlay-masking detection in action - for visually
demonstrating "scoreboard being detected" on screen during a recording.
Not part of the pipeline itself (see pipeline.py for that).

Usage:
    python demo_visualize.py
Press any key to advance between windows, Esc to quit early.
"""
import cv2

from grid import load_layout, cells_for_layout
from layout_selector import LayoutSelector
import overlay as overlay_mod


def show(title, img, max_w=1280):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyWindow(title)
    return key


def main():
    selector = LayoutSelector(config_dir="../config", calibration_frame="../data/frames/frame_000001.png")

    # 1) grid detection on a clean frame
    frame1 = cv2.imread("../data/frames/frame_000001.png")
    layout, state = selector.select(frame1)
    overlay_img = frame1.copy()
    for cell in cells_for_layout(layout):
        x1, y1, x2, y2 = cell.bbox
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if show("1) Calibrated grid detected on input frame - press any key", overlay_img) == 27:
        return

    # 2) overlay (pin-fall animation) detection + masking on a frame that has it
    frame22 = cv2.imread("../data/frames/frame_000022.png")
    layout22, state22 = selector.select(frame22)
    obscured = overlay_mod.obscured_blocks(frame22, layout22, state22)
    bounds = overlay_mod.block_bounds(layout22)
    x1, x2 = int(bounds["col6_left"]), int(bounds["col10_right"])
    vis = frame22.copy()
    for player in obscured:
        y1, y2 = bounds["blocks"][player]
        cv2.rectangle(vis, (x1, int(y1)), (x2, int(y2)), (0, 0, 255), 4)
        cv2.putText(vis, "OVERLAY DETECTED - MASKED", (x1 + 10, int(y1) + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    print(f"Blocks masked this frame: {sorted(obscured)}")
    show("2) Transient overlay detected + masked - press any key to close", vis)


if __name__ == "__main__":
    main()
