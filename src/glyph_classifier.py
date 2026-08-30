"""
Template-matching glyph recognizer, built specifically because Tesseract
mis-reads several digits in this broadcast graphic's stylized font (e.g. it
consistently reads a very clear '9' as nothing at all, across every PSM mode -
verified against real footage, not a preprocessing issue).

Since the vocabulary is tiny (0-9, X, /, -) and the font is fixed (it's
rendered by broadcast software, not photographed), per-glyph template
matching is far more reliable than general-purpose OCR here.

Templates live as small PNGs under config/templates/<kind>/<label>/*.png,
where <kind> is 'score' or 'symbols' (the two row types use different font
sizes) and <label> is the character (with 'SLASH' / 'DASH' standing in for
'/' and '-', since those aren't valid filename/dirname characters on
Windows).
"""
from pathlib import Path

import cv2
import numpy as np

CANON_SIZE = (40, 56)  # (w, h) every glyph crop is normalized to before matching

LABEL_TO_DIRNAME = {"/": "SLASH", "-": "DASH"}
DIRNAME_TO_LABEL = {v: k for k, v in LABEL_TO_DIRNAME.items()}


def _dirname_for_label(label):
    return LABEL_TO_DIRNAME.get(label, label)


def _label_for_dirname(dirname):
    return DIRNAME_TO_LABEL.get(dirname, dirname)


def segment_glyphs(binarized, min_area_frac=0.008, border_margin=25, thin_frac=0.21, fill_thresh=0.30):
    """binarized: dark text (0) on light background (255), as produced by
    recognize.preprocess(). Returns glyph crops (grayscale, dark-on-light)
    ordered left-to-right, each as (x_center, crop).

    min_area_frac is relative to the (possibly upscaled) image area, so it
    stays valid regardless of preprocess()'s upscale factor.

    Two distinct kinds of border-adjacent noise show up here, so they need
    two distinct filters (a single one catches one at the cost of the other):
      - small solid slivers (grid-line/anti-aliasing bleed from crop padding):
        thin in their narrow dimension.
      - the full cell's colored-block outline (e.g. the T row's yellow/white
        boundary), which shows up as one component spanning nearly the whole
        crop in BOTH directions with a hollow, low-fill interior.
    A component is only discarded if it touches the border AND matches one of
    those two shapes - border-touching alone isn't enough, since a real glyph
    can legitimately reach the edge (e.g. a '/' whose diagonal stroke reaches
    a corner - which also rules out fill ratio alone, since a diagonal stroke
    is sparse in its bbox too, but doesn't span both dimensions fully)."""
    inv = cv2.bitwise_not(binarized)  # text becomes foreground (255) for connected components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)

    h, w = binarized.shape
    min_area = min_area_frac * (h * w)
    thin_thresh = thin_frac * min(h, w)
    glyphs = []
    for i in range(1, n):  # skip background label 0
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        touches_border = (x <= border_margin or y <= border_margin
                           or x + cw >= w - border_margin or y + ch >= h - border_margin)
        fill_ratio = area / (cw * ch)
        is_thin_sliver = min(cw, ch) < thin_thresh
        is_hollow_outline = cw >= 0.85 * w and ch >= 0.85 * h and fill_ratio < fill_thresh
        # a grid line spanning the full cell height/width but with enough
        # anti-aliasing halo to dodge the thin-sliver check above
        is_full_line = fill_ratio < 0.5 and (
            (ch >= 0.85 * h and cw < 0.3 * w) or (cw >= 0.85 * w and ch < 0.3 * h)
        )
        # catch-all for border artifacts that don't match either specific
        # shape above - e.g. an L-shaped corner bleed (verified on
        # frame_000081's J/score/col4: a right+bottom border-line pair
        # merged into one component, 40% width and full height, sparse
        # enough (fill ~0.04) to be unambiguously not a glyph - the lowest
        # fill measured for any real glyph here is '/' at ~0.16)
        is_very_sparse = fill_ratio < 0.10
        if touches_border and (is_thin_sliver or is_hollow_outline or is_full_line or is_very_sparse):
            continue
        glyphs.append((centroids[i][0], binarized[y:y + ch, x:x + cw]))

    glyphs.sort(key=lambda g: g[0])
    return [g[1] for g in glyphs]


def split_component(crop, n):
    """Split a connected-component crop containing n touching glyphs (common
    in the score row's larger font, where digit kerning makes neighbors
    overlap by a pixel or two) using valley-finding on the column ink profile.

    The valley search covers the whole interior width (not just a window
    around the n-way-even split point) - a narrow window assumes the glyphs
    are roughly equal width, which fails for pairs involving '1' (much
    narrower than other digits): verified on a "31" blob whose true valley
    sits at ~70% of the width, comfortably outside a +/-40%-of-center window,
    which instead picked a spurious local minimum near the window's edge and
    cut '3' itself in half."""
    h, w = crop.shape
    if n <= 1:
        return [crop]
    if n != 2:
        raise NotImplementedError("split_component only handles n=2 (the only case ever needed here)")
    ink = (255 - crop.astype(np.int32)).sum(axis=0)  # per-column ink amount
    margin = max(1, int(w * 0.25))
    lo, hi = margin, w - margin
    split = int(w / 2) if hi <= lo else lo + int(np.argmin(ink[lo:hi]))
    return [crop[:, :split], crop[:, split:]]


def _normalize(glyph_crop):
    """Pad to square-ish aspect then resize to CANON_SIZE, so stroke width /
    proportions are comparable across different source font sizes."""
    h, w = glyph_crop.shape
    scale = min(CANON_SIZE[0] / w, CANON_SIZE[1] / h) * 0.85
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(glyph_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full(CANON_SIZE[::-1], 255, dtype=np.uint8)
    ox, oy = (CANON_SIZE[0] - new_w) // 2, (CANON_SIZE[1] - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


class GlyphClassifier:
    def __init__(self, template_dir="config/templates"):
        self.template_dir = Path(template_dir)
        self.templates = {"score": {}, "symbols": {}, "lane": {}}
        self.median_width = {"score": None, "symbols": None, "lane": None}
        self._load()

    def _load(self):
        for kind in ("score", "symbols", "lane"):
            kind_dir = self.template_dir / kind
            if not kind_dir.exists():
                continue
            widths = []
            for label_dir in kind_dir.iterdir():
                label = _label_for_dirname(label_dir.name)
                imgs = []
                for f in label_dir.glob("*.png"):
                    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imgs.append(_normalize(img))
                        widths.append(img.shape[1])
                if imgs:
                    self.templates[kind][label] = imgs
            if widths:
                self.median_width[kind] = float(np.median(widths))

    def add_template(self, kind, label, glyph_crop, save=True, index=None):
        self.templates[kind].setdefault(label, []).append(_normalize(glyph_crop))
        if save:
            out_dir = self.template_dir / kind / _dirname_for_label(label)
            out_dir.mkdir(parents=True, exist_ok=True)
            if index is None:
                index = len(list(out_dir.glob("*.png")))
            cv2.imwrite(str(out_dir / f"{index}.png"), glyph_crop)

    def classify(self, glyph_crop, kind):
        norm = _normalize(glyph_crop)
        # fall back to the other kind's templates if this kind has none yet
        # (score/symbols share the same font family, just a different point size)
        bank = self.templates[kind] or self.templates["score"] or self.templates["symbols"]
        best_label, best_score = None, -2.0
        for label, templates in bank.items():
            for tmpl in templates:
                res = cv2.matchTemplate(norm, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(res[0, 0])
                if score > best_score:
                    best_score, best_label = score, label
        return best_label, best_score

    def recognize_lane_best_of(self, binarized_variants):
        """The lane number's font is loopy enough that connected-component
        segmentation splits a single digit into disconnected pieces (verified
        on frame_000001's '6': an 8-connectivity gap between the diagonal
        stroke and the loop splits it into two components, which would
        otherwise both get classified as separate digits, e.g. '66').
        Assumes a single-digit lane number throughout the video (true here) -
        merges ALL segmented pieces into one bounding region rather than
        trying to tell a connectivity gap apart from a genuine second digit,
        which would need adjusting for a multi-digit lane number."""
        best_text, best_conf = "", -2.0
        for binarized in binarized_variants:
            glyphs = segment_glyphs(binarized)
            if not glyphs:
                continue
            h, w = binarized.shape
            # segment_glyphs only returns crops, not coords - re-derive boxes to merge them
            inv = cv2.bitwise_not(binarized)
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
            boxes = [stats[i] for i in range(1, n)
                     if stats[i][4] >= 0.008 * (h * w)]
            if not boxes:
                continue
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            merged = binarized[y1:y2, x1:x2]
            label, conf = self.classify(merged, "lane")
            if label is not None and conf > best_conf:
                best_text, best_conf = label, conf
        return best_text

    def recognize_cell_best_of(self, binarized_variants, kind):
        """Try recognize_cell against several binarized versions of the same
        cell (see recognize.preprocess_variants) and keep whichever result
        the classifier is most confident about overall. No single
        binarization setting handles every frame's noise/gradient tradeoff,
        but per-glyph template-match confidence is a reliable signal for
        which attempt actually worked on this particular cell."""
        best_text, best_conf = "", -2.0
        for binarized in binarized_variants:
            text, conf = self.recognize_cell(binarized, kind, return_confidence=True)
            if conf > best_conf:
                best_text, best_conf = text, conf
        return best_text

    def _best_split(self, glyph, kind, margin_frac=0.15, step=3):
        """Find the split point that gives the best classification result,
        rather than trusting the ink-profile valley (split_component) to mark
        the true character boundary. Ink-valley splitting fails when a
        digit's own internal shape has a lower dip than the actual boundary -
        verified on a "31" blob, where '3''s two lobes create an internal dip
        deeper than the real 3|1 valley, so every ink-based margin either
        caught that false dip or (widened further) the character's own edge
        taper; no single margin worked for both this and normal cases.
        Classification confidence doesn't have that ambiguity: only the
        genuine split point classifies both halves as digits at once.

        Candidates are ranked by SUM (not min) of the two halves' confidence.
        Verified this matters on a "28" blob: the true split ('2' 0.93, '8'
        0.77) lost to a wrong one one column over ('2' 0.78, '3' 0.79) under
        a min-based ranking, since '8' and '3' share enough of their upper
        curve that a badly-placed cut can score deceptively well on both
        sides at once - but sum correctly favors the split with one very
        confident half (1.70) over two merely-decent ones (1.57)."""
        h, w = glyph.shape
        margin = max(1, int(w * margin_frac))
        best = None
        for x in range(margin, w - margin, step):
            l_label, l_conf = self.classify(glyph[:, :x], kind)
            r_label, r_conf = self.classify(glyph[:, x:], kind)
            if best is None or l_conf + r_conf > best[0]:
                best = (l_conf + r_conf, [(l_label, l_conf), (r_label, r_conf)])
        return best  # (confidence sum, [(label, conf), (label, conf)]) or None

    def _recognize_from_glyphs(self, glyphs, kind, split_confidence):
        """A split is always ATTEMPTED for score-kind glyphs (not gated on
        whole-blob confidence) - a wrong whole-blob single-digit read can
        score deceptively high, above any reasonable confidence gate:
        verified on a "27" blob that read as a lone '2' at 0.89 confidence,
        well past a 0.85 gate, silently skipping the split that would have
        found the correct "27" (0.77). Whether the split is KEPT is decided
        by comparing it against the whole-blob result directly (min part
        confidence beats whole-blob confidence) rather than an absolute bar -
        verified against frame_000081's "20", where the correct split scores
        ('2' 0.97, '0' 0.72) both cleared an (also since-removed) fixed 0.85
        threshold check for '2' but not '0', rejecting a split that was still
        far better than the whole-blob misread ('9' at 0.47) it fell back to.

        (An earlier version also gated splitting on blob width, based on a
        misreading of frame_000081's J/score/col4 "31": what looked like a
        false-positive split of "a lone 3" was actually the genuine "31"
        blob being split correctly - the character that looked spurious was
        coming from a *separate*, unrelated border-noise component that
        segment_glyphs wasn't filtering out. Fixed at the source (see
        segment_glyphs' is_very_sparse check) instead, since the width gate
        was blocking legitimate splits of exactly this kind of narrow
        two-digit blob.)

        split_confidence is unused here now (kept as a parameter only because
        recognize_cell() also uses it to gate the unrelated symbols-kind
        erosion retry below) - score splitting no longer needs a gate since
        every attempt is already validated against the whole-blob result."""
        chars = []
        confidences = []
        median_w = self.median_width.get(kind)
        for g in glyphs:
            label, conf = self.classify(g, kind)
            if kind == "score":
                best = self._best_split(g, kind)
                # a blob unambiguously too wide for one digit (verified:
                # every single digit measured 143-172px vs. this font's
                # ~159.5 median, while every genuine merge measured 300+) is
                # accepted on width alone, bypassing the confidence
                # comparison - which a wrong-but-confident whole-blob read
                # can still win: verified on a "27" blob that read as a lone
                # '2' at 0.89, comfortably beating the correct split's weaker
                # half ('7' at 0.74) despite being flat wrong. Narrower blobs
                # (single wide digit vs. a tight merge) stay on the
                # confidence comparison, where that ambiguity is genuine.
                is_unambiguously_wide = median_w and g.shape[1] >= 1.7 * median_w
                if best is not None and (is_unambiguously_wide or min(s for _, s in best[1]) > conf):
                    for l, s in best[1]:
                        chars.append(l)
                        confidences.append(s)
                    continue
            if label is not None:
                chars.append(label)
                confidences.append(conf)
        text = "".join(chars)
        overall_conf = min(confidences) if confidences else 1.0  # nothing found = nothing to doubt
        return text, overall_conf

    def recognize_cell(self, binarized, kind, split_confidence=0.85, return_confidence=False):
        """Only the score row's larger font has observed touching-digit
        merges (e.g. "20", "48" rendering as one connected component). Digit
        width varies too much (a lone '1' vs a lone '8') for a width-based
        merge heuristic to be reliable - it was tried and mis-split
        legitimately wide single digits. Instead: classify the whole
        component first: a real single glyph matches its template with very
        high confidence (~1.0 in practice). Only if that confidence is low
        (ambiguous/no good match - the signature of two touching digits
        being forced into one classification) do we try splitting in two and
        check whether BOTH halves then match confidently; if so, the split
        wins. Symbol-row glyphs (X, /, -, digits) don't need this - natural
        segmentation already separates them correctly.

        Symbol-row cells have a second, unrelated failure mode: the graphic
        circles whichever roll was just thrown, and that ring merges with
        (and sometimes bridges across to a neighboring) glyph into one
        connected component - verified on frame_000001's V/col4 ('8' with a
        circle around it, next to '1'): normal segmentation produces one
        giant blob spanning both characters, which classifies as garbage. A
        light erosion breaks the thin ring/bridge apart while leaving the
        thicker digit strokes intact (verified: the eroded blob then
        classifies correctly as '8' at 0.54 confidence, and '1' separates out
        at 0.92) - so when confidence is low, symbol cells also get an
        erosion retry, keeping whichever attempt scored higher overall."""
        glyphs = segment_glyphs(binarized)
        text, conf = self._recognize_from_glyphs(glyphs, kind, split_confidence)

        if kind == "symbols" and conf < split_confidence:
            inv = cv2.bitwise_not(binarized)
            eroded = cv2.erode(inv, np.ones((5, 5), np.uint8), iterations=2)
            eroded_glyphs = segment_glyphs(cv2.bitwise_not(eroded))
            eroded_text, eroded_conf = self._recognize_from_glyphs(eroded_glyphs, kind, split_confidence)
            if eroded_conf > conf:
                text, conf = eroded_text, eroded_conf

        if not return_confidence:
            return text
        return text, conf
