"""
Step 5: Sanity-check OCR output against bowling scoring rules, so a single
misread cell doesn't silently corrupt a player's scorecard.

Kept deliberately simple for now (monotonicity + plausible-delta checks).
Exact per-frame pinfall reconciliation is harder than it looks because
broadcast graphics show *provisional* running totals that get revised once
strike/spare bonus rolls land - once we have real footage we can tell how
this particular graphic handles that lag and tighten these rules.
"""
from typing import List, Optional

MAX_FRAME_DELTA = 30  # two bonus strikes after a strike in frame 10, worst case


def validate_totals(totals: List[Optional[int]]) -> List[str]:
    """totals: running total per frame, in order, None where not yet OCR'd/blank."""
    issues = []
    prev = None
    prev_idx = None
    for i, t in enumerate(totals, start=1):
        if t is None:
            continue
        if prev is not None:
            delta = t - prev
            if delta < 0:
                issues.append(f"frame {i}: total {t} is lower than frame {prev_idx}'s {prev} (non-monotonic)")
            elif delta > MAX_FRAME_DELTA:
                issues.append(f"frame {i}: total {t} jumped by {delta} from frame {prev_idx} (implausible, likely OCR error)")
        prev, prev_idx = t, i
    return issues


def is_plausible_symbol(symbol: str) -> bool:
    return symbol in {"", "X"} or symbol.isdigit() or "/" in symbol or "-" in symbol


def validate_ttl(scores: List[Optional[int]], ttl: Optional[int],
                  symbols: Optional[List[Optional[str]]] = None) -> List[str]:
    """TTL usually mirrors the player's most-recently-completed frame's
    running total (verified: frame_000001's J row shows frame-4 total 31 and
    TTL 31) - but not always: when the next frame has a roll already in
    progress, TTL shows a *provisional* total that speculatively includes
    it, before that frame's own score column gets filled in. Verified two
    forms of this against real footage: a strike with the frame still
    unscored (frame_000109's J: frame-4 total 31, frame-5 symbol 'X', TTL 41
    = 31+10) and a single partial roll of an open frame (frame_000157's V:
    frame-4 total 28, frame-5 symbol '9', TTL 37 = 28+9). Both are legitimate
    displayed values, not OCR errors, so a mismatch is only flagged if it
    doesn't match either the plain or the provisional total."""
    if ttl is None:
        return []
    last_idx = next((i for i in range(len(scores) - 1, -1, -1) if scores[i] is not None), None)
    if last_idx is None:
        return []
    last_known = scores[last_idx]
    if ttl == last_known:
        return []
    if symbols is not None and last_idx + 1 < len(symbols):
        nxt = symbols[last_idx + 1]
        provisional = None
        if nxt == "X":
            provisional = last_known + 10
        elif nxt is not None and len(nxt) == 1 and nxt.isdigit():
            provisional = last_known + int(nxt)
        if provisional == ttl:
            return []
    return [f"TTL {ttl} doesn't match latest frame total {last_known}"]
