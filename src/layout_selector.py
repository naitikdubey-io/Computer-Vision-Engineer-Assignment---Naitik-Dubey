"""
Two things this handles, both discovered against the real source video rather
than assumed up front:

1. The recording isn't continuous scoreboard footage - it cuts to promotional
   bumpers, cartoons, and a channel ID between scoreboard segments. Frames
   that aren't showing the scoreboard need to be skipped before attempting
   any grid extraction on them.

2. The scoreboard graphic itself isn't pixel-static: when a player's row
   becomes the "active bowler" (highlighted red/yellow instead of blue), the
   row block below the header grows, shifting every row after it down. The
   header itself (lane number, column labels) never moves. So a single fixed
   calibration only covers one layout state - this picks between pre-
   calibrated states per frame instead of assuming one fixed grid.

"None", "J", and "V" states have been calibrated (the only ones seen in the
source video - P never becomes the active bowler in this 58s clip, confirmed
by scanning every frame). Add a config/layout_P.json + a third anchor check
below if a future video shows P highlighted.

Anchor checks are SEQUENTIAL, not independent: a row's highlight-anchor point
is only valid to read once every row *before* it has been confirmed not
highlighted. When J's row grows (highlighted), it pushes V's row down too, so
V's anchor point (which assumes V starts where it does in the "none" state)
lands inside the tail of J's own (now taller) block instead of V's - verified
directly against frame_000081 (J highlighted), where the V-anchor pixel reads
pure white (J's own score-row background) rather than V's expected blue.
Checking J first and only falling through to V if J isn't highlighted avoids
this; the same reasoning would apply to a P check falling through from V.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from grid import load_layout

# Fixed points just below each row's start in the "none" state (valid to
# check only once every earlier row is confirmed not highlighted - see
# module docstring). BGR red-channel value: ~40-56 when normal blue, ~250-255
# when highlighted pale yellow/white - verified against frame_000001.png (no
# highlight), frame_000081.png (J highlighted), frame_000109.png (V
# highlighted).
J_ANCHOR_POINT = (300, 160)
V_ANCHOR_POINT = (300, 330)
HIGHLIGHT_RED_THRESHOLD = 150

# Header region used both to locate the graphic (dy=0 always, verified) and
# to gate out non-scoreboard frames via template-match confidence.
HEADER_TEMPLATE_BOX = (0, 0, 900, 140)  # x, y, w, h in calibration-frame coords
SCOREBOARD_MATCH_THRESHOLD = 0.5


class LayoutSelector:
    def __init__(self, config_dir="../config", calibration_frame="../data/frames/frame_000001.png"):
        self.layouts = {
            "none": load_layout(f"{config_dir}/layout.json"),
            "J": load_layout(f"{config_dir}/layout_J.json"),
            "V": load_layout(f"{config_dir}/layout_V.json"),
        }
        calib_img = cv2.imread(calibration_frame)
        x, y, w, h = HEADER_TEMPLATE_BOX
        self.header_template = cv2.cvtColor(calib_img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    def is_scoreboard_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.header_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= SCOREBOARD_MATCH_THRESHOLD

    def active_state(self, frame):
        """Returns 'none', 'J', or 'V' - the row-highlight state - without
        the is_scoreboard_frame gate (callers that already know it's a
        scoreboard frame, e.g. the overlay detector, can skip that check)."""
        jx, jy = J_ANCHOR_POINT
        if frame[jy, jx, 2] >= HIGHLIGHT_RED_THRESHOLD:
            return "J"
        vx, vy = V_ANCHOR_POINT
        if frame[vy, vx, 2] >= HIGHLIGHT_RED_THRESHOLD:
            return "V"
        return "none"

    def select(self, frame):
        """Returns (layout dict, state name), or (None, None) if this isn't a
        scoreboard frame at all."""
        if not self.is_scoreboard_frame(frame):
            return None, None
        state = self.active_state(frame)
        return self.layouts[state], state
