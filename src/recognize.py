"""
Step 4: Turn a cropped cell image into text.

MVP uses Tesseract with a restricted character whitelist, since the vocabulary
per cell is tiny (digits, X, /, -). If accuracy isn't good enough once we test
against real footage, swap this module for a template-matching / small-CNN
glyph classifier trained on the broadcast's actual font - the rest of the
pipeline doesn't need to change, since everything downstream just consumes
`recognize_cell(img) -> str`.
"""
import cv2
import numpy as np
import pytesseract

from tool_paths import get_tesseract

pytesseract.pytesseract.tesseract_cmd = get_tesseract()

SCORE_WHITELIST = "0123456789"
SYMBOL_WHITELIST = "0123456789X/-"


def _binarize(gray, clip_limit, blur_ksize):
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # normalize polarity to dark text (0) on light background (255): text is
    # reliably the minority of pixels in a cell, whichever polarity the
    # source row uses (white-on-blue vs the team-total row's dark-on-light)
    if np.count_nonzero(thresh == 0) > np.count_nonzero(thresh == 255):
        thresh = cv2.bitwise_not(thresh)
    return thresh


def preprocess(cell_bgr, upscale=4, clip_limit=2.0, blur_ksize=(3, 3)):
    """Cell backgrounds are a blue (or yellow/white, for the team-total row)
    gradient, not a flat color - a plain global Otsu threshold can get
    fooled when the gradient itself crosses the chosen cutoff (e.g. the
    darker end of the gradient lands on the same side as the text), which
    inverts polarity for part of the cell (verified against real footage:
    frame 87's V-row cell). CLAHE equalizes contrast within local tiles
    first, which flattens the gradient's effect while still leaving a clean
    global bimodal split for Otsu to find afterward.

    clip_limit/blur_ksize are exposed because no single setting handles
    every frame: a stronger clip_limit (more local contrast) is needed to
    beat the background gradient, but on a blurrier/more-compressed frame
    (verified on frame 81, which measured notably blurrier than frame 1) the
    same setting amplifies compression noise into jagged edges. See
    preprocess_variants() / GlyphClassifier's confidence-based selection,
    which picks whichever setting works for a given cell rather than
    committing to one globally."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return _binarize(gray, clip_limit, blur_ksize)


# (clip_limit, blur_ksize) pairs to try, ordered strongest-contrast-first.
PREPROCESS_VARIANTS = [
    (2.0, (3, 3)),  # handles the background-gradient case
    (1.0, (5, 5)),  # gentler, handles noisier/blurrier frames
]


def preprocess_variants(cell_bgr, upscale=4):
    return [preprocess(cell_bgr, upscale, clip, blur) for clip, blur in PREPROCESS_VARIANTS]


def is_blank(binarized, fg_ratio_threshold=0.01):
    """A cell with almost no foreground pixels is empty (frame not played yet)."""
    fg = np.count_nonzero(binarized < 128)
    return (fg / binarized.size) < fg_ratio_threshold


# (blank std, text std) differ enough between the two row kinds that neither
# a single global threshold nor one calibrated on only one kind works for
# both - verified directly: blank measured ~36 for symbols cells vs. ~22-24
# for score cells, while text measured ~57-62 for symbols vs. ~40-44 for
# score. The two kinds' ranges overlap across that gap (blank-symbols ~36
# sits inside text-score's ~40-44 neighborhood), so this can't be one number.
BLANK_STD_THRESHOLD = {"symbols": 47, "score": 32}


def is_likely_blank(cell_bgr, kind="symbols"):
    """Cheaper, more reliable blank check than is_blank() alone: run BEFORE
    preprocess(), on the raw crop. A genuinely blank cell (just the
    background gradient, frame not yet played) has low pixel std regardless
    of CLAHE; real text has much higher raw contrast. is_blank() alone isn't
    enough because CLAHE can manufacture fake high-contrast edges out of a
    blank cell's subtle gradient - verified on frame_000022's P/col7 (truly
    empty), where CLAHE turned the gradient into a fake horizontal streak
    that survived noise filtering and got misread as '-'."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    return gray.std() < BLANK_STD_THRESHOLD[kind]


def recognize_cell(cell_bgr, kind="symbols"):
    """kind: 'symbols' (roll marks, e.g. 'X', '5-', '8/') or 'score' (running total)."""
    binarized = preprocess(cell_bgr)
    if is_blank(binarized):
        return ""

    whitelist = SCORE_WHITELIST if kind == "score" else SYMBOL_WHITELIST
    config = f'--psm 8 -c tessedit_char_whitelist={whitelist}'
    text = pytesseract.image_to_string(binarized, config=config)
    return text.strip()
