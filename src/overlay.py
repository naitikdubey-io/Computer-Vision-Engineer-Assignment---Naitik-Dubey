"""
The source video pops up a transient "which pins fell" animation over part of
the scoreboard for a few seconds after most rolls - always over columns 6-10,
but NOT at a fixed vertical position: it tracks whichever row block just got
action (verified directly: frame_000022 shows it over J/V's rows, frame_000160
shows the same graphic lower down, over P/T's rows). A fixed bounding box
can't catch both, so detection instead checks each row-block (J, V, P, T)
independently against its own known-clean color baseline, using only the
columns 6-10 slice of that block (the only region ever observed affected).

The overlay's navy background reads much lower on the green channel than any
legitimate cell background here - verified against every background this
graphic uses: plain cyan/blue (~170-180), the active-bowler's own
yellow/white highlight (~248), and the team-total row's two different styles
(~251 in the "none" state's permanent yellow/white styling, ~147 in any
highlighted state, where it reverts to plain blue-ish) - so a per-block
baseline (which one applies depends on layout state and whether that block is
the currently-highlighted one) with a 20-point margin cleanly separates
"overlay present" from every clean case measured, with room to spare (closest
clean gap ~30pts info T; worst observed overlay reading vs. baseline was a
~30-65pt drop in every confirmed case).

A per-CELL version of this check was tried first and rejected: individual
cells are dominated by their own text content, not the background, so the
signal was too noisy (clean cells measured green means as low as ~105,
indistinguishable from overlay-covered ones).
"""
BASELINE_GREEN = {
    ("none", "J"): 176.6, ("none", "V"): 177.1, ("none", "P"): 169.4, ("none", "T"): 251.5,
    ("J", "J"): 248.0, ("J", "V"): 172.9, ("J", "P"): 167.7, ("J", "T"): 147.5,
    ("V", "J"): 176.6, ("V", "V"): 251.7, ("V", "P"): 169.4, ("V", "T"): 147.5,
}
DROP_THRESHOLD = 20  # baseline - actual >= this => flag as obscured


def block_bounds(layout):
    rx, ry, rw, rh = layout["roi"]
    row_px = [ry + f * rh for f in layout["row_bounds_frac"]]
    col_px = [rx + f * rw for f in layout["col_bounds_frac"]]
    return {
        "blocks": {"J": (row_px[0], row_px[2]), "V": (row_px[2], row_px[4]),
                   "P": (row_px[4], row_px[6]), "T": (row_px[6], row_px[8])},
        "col6_left": col_px[5],
        "col10_right": col_px[9],
    }


def obscured_blocks(frame, layout, state):
    """Returns the set of player letters ('J'/'V'/'P'/'T') whose columns
    6-10 are currently covered by the pin overlay in this frame."""
    bounds = block_bounds(layout)
    x1, x2 = int(bounds["col6_left"]), int(bounds["col10_right"])
    obscured = set()
    for player, (y1, y2) in bounds["blocks"].items():
        region = frame[int(y1):int(y2), x1:x2]
        actual = float(region[:, :, 1].mean())
        baseline = BASELINE_GREEN[(state, player)]
        if baseline - actual >= DROP_THRESHOLD:
            obscured.add(player)
    return obscured


def cell_is_obscured(cell, obscured_players):
    return cell.row_label["player"] in obscured_players and cell.col_label not in ("1", "2", "3", "4", "5", "TTL")
