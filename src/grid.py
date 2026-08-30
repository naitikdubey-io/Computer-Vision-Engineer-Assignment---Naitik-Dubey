"""
Shared helpers: load a calibrated layout.json and turn it into pixel-accurate
cell bounding boxes for a given frame.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cell:
    row_label: dict   # {"player": "J", "type": "symbols"}
    col_label: str     # "1".."10" or "TTL"
    bbox: tuple         # (x1, y1, x2, y2) in full-frame pixel coords


def load_layout(path="config/layout.json") -> dict:
    return json.loads(Path(path).read_text())


def cells_for_layout(layout: dict):
    """Yields Cell objects covering every (row, col) in the calibrated grid."""
    rx, ry, rw, rh = layout["roi"]
    col_bounds = layout["col_bounds_frac"]
    row_bounds = layout["row_bounds_frac"]
    col_labels = layout["col_labels"]
    row_labels = layout["row_labels"]

    col_px = [rx + f * rw for f in col_bounds]
    row_px = [ry + f * rh for f in row_bounds]

    for ri, row_label in enumerate(row_labels):
        y1, y2 = row_px[ri], row_px[ri + 1]
        for ci, col_label in enumerate(col_labels):
            x1, x2 = col_px[ci], col_px[ci + 1]
            yield Cell(
                row_label=row_label,
                col_label=col_label,
                bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )


def crop_cell(frame, cell: Cell, pad: int = 2):
    x1, y1, x2, y2 = cell.bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return frame[y1:y2, x1:x2]
