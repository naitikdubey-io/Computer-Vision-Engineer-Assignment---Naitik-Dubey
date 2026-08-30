"""
One-time bootstrap: extract labeled glyph templates from frame_000001, whose
values we've verified by eye against the source video. Populates
config/templates/<score|symbols>/<label>/*.png for glyph_classifier.py.

Skips cells with known segmentation hazards (e.g. the circled "current roll"
digit) - those can be added by hand later if needed.
"""
import cv2

from grid import load_layout, cells_for_layout, crop_cell
from recognize import preprocess
from glyph_classifier import GlyphClassifier, segment_glyphs, split_component

# (player, kind, col) -> ground-truth characters, left to right.
# Verified against data/frames/frame_000001.png by inspection.
GROUND_TRUTH = {
    ("J", "symbols", "1"): ["X"],
    ("J", "symbols", "2"): ["5", "-"],
    ("J", "symbols", "3"): ["-", "7"],
    ("J", "symbols", "4"): ["4", "-"],
    ("V", "symbols", "1"): ["8", "-"],
    ("V", "symbols", "2"): ["3", "-"],
    ("V", "symbols", "3"): ["7", "1"],
    # V col4 skipped: circled digit, handled separately
    ("P", "symbols", "1"): ["X"],
    ("P", "symbols", "2"): ["4", "/"],
    ("P", "symbols", "3"): ["9", "-"],
    ("P", "symbols", "4"): ["6", "-"],
    ("T", "symbols", "1"): ["6", "1"],
    ("T", "symbols", "2"): ["1", "/"],
    ("T", "symbols", "3"): ["8", "-"],

    ("J", "score", "1"): ["1", "5"],
    ("J", "score", "2"): ["2", "0"],
    ("J", "score", "3"): ["2", "7"],
    ("J", "score", "4"): ["3", "1"],
    ("V", "score", "1"): ["8"],
    ("V", "score", "2"): ["1", "1"],
    ("V", "score", "3"): ["1", "9"],
    ("V", "score", "4"): ["2", "8"],
    ("P", "score", "1"): ["2", "0"],
    ("P", "score", "2"): ["3", "9"],
    ("P", "score", "3"): ["4", "8"],
    ("P", "score", "4"): ["5", "4"],
    ("T", "score", "1"): ["7"],
    ("T", "score", "2"): ["2", "5"],
    ("T", "score", "3"): ["3", "3"],
}


def main():
    layout = load_layout("../config/layout.json")
    img = cv2.imread("../data/frames/frame_000001.png")
    clf = GlyphClassifier(template_dir="../config/templates")

    mismatches = []
    for cell in cells_for_layout(layout):
        key = (cell.row_label["player"], cell.row_label["type"], cell.col_label)
        if key not in GROUND_TRUTH:
            continue
        expected = GROUND_TRUTH[key]
        crop = crop_cell(img, cell)
        binarized = preprocess(crop)
        glyphs = segment_glyphs(binarized)

        if len(glyphs) != len(expected):
            # common case: touching digits in the score font merged into one
            # connected component - split it back apart since we know how many
            # characters this cell should contain
            if 0 < len(glyphs) < len(expected):
                widest_idx = max(range(len(glyphs)), key=lambda i: glyphs[i].shape[1])
                need = len(expected) - len(glyphs) + 1
                parts = split_component(glyphs[widest_idx], need)
                glyphs = glyphs[:widest_idx] + parts + glyphs[widest_idx + 1:]

            if len(glyphs) != len(expected):
                mismatches.append((key, expected, len(glyphs)))
                continue

        for glyph_crop, label in zip(glyphs, expected):
            clf.add_template(cell.row_label["type"], label, glyph_crop)

    print(f"Loaded templates for kinds: "
          f"score={ {k: len(v) for k, v in clf.templates['score'].items()} }, "
          f"symbols={ {k: len(v) for k, v in clf.templates['symbols'].items()} }")
    if mismatches:
        print("\nSegmentation mismatches (glyph count didn't match expected - skipped):")
        for key, expected, got in mismatches:
            print(f"  {key}: expected {len(expected)} glyphs {expected}, segmented {got}")


if __name__ == "__main__":
    main()
