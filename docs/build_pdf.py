"""
Builds docs/documentation.pdf - the assignment's required documentation:
screenshots of an input frame, the code running, the detected scoreboard,
and the extracted output, each with a short explanation.

Run from the docs/ directory: python build_pdf.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SCREENSHOTS = Path(__file__).parent / "screenshots"
OUT = Path(__file__).parent / "documentation.pdf"

NAVY = colors.HexColor("#16202E")
MUTED = colors.HexColor("#5A6B80")
ACCENT = colors.HexColor("#1F5FBF")
ACCENT_STRONG = colors.HexColor("#123A73")
BORDER = colors.HexColor("#D7DEE8")
WASH = colors.HexColor("#EEF2F7")

pdfmetrics.registerFont(TTFont("PlexSans", r"C:\Windows\Fonts\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("PlexSans-Bold", r"C:\Windows\Fonts\segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("Mono", r"C:\Windows\Fonts\consola.ttf"))

styles = {
    "title": ParagraphStyle("title", fontName="PlexSans-Bold", fontSize=25,
                             leading=30, textColor=NAVY, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="PlexSans", fontSize=11.5,
                                leading=17, textColor=MUTED, spaceAfter=4),
    "eyebrow": ParagraphStyle("eyebrow", fontName="Mono", fontSize=9,
                               leading=12, textColor=ACCENT, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="PlexSans-Bold", fontSize=16.5,
                          leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=2),
    "secnum": ParagraphStyle("secnum", fontName="Mono", fontSize=10,
                              leading=14, textColor=ACCENT),
    "body": ParagraphStyle("body", fontName="PlexSans", fontSize=10.3,
                            leading=15.5, textColor=NAVY, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="PlexSans", fontSize=9,
                               leading=13, textColor=MUTED, spaceBefore=4, spaceAfter=2),
    "footer": ParagraphStyle("footer", fontName="PlexSans", fontSize=8.5,
                              leading=12, textColor=MUTED),
}


def section_header(num, title_text):
    t = Table(
        [[Paragraph(num, styles["secnum"]), Paragraph(title_text, styles["h2"])]],
        colWidths=[0.35 * inch, 6.0 * inch],
    )
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def framed_image(path, max_w, max_h):
    img = RLImage(str(path))
    ratio = img.imageWidth / img.imageHeight
    w, h = max_w, max_w / ratio
    if h > max_h:
        h = max_h
        w = max_h * ratio
    img.drawWidth, img.drawHeight = w, h
    tbl = Table([[img]], colWidths=[w + 12])
    tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, BORDER),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def rule():
    return HRFlowable(width="100%", thickness=1, color=BORDER, spaceBefore=18, spaceAfter=18)


story = []

# ---------- Cover ----------
story.append(Paragraph("DOCUMENTATION &middot; COMPUTER VISION ASSIGNMENT", styles["eyebrow"]))
story.append(Paragraph("Scoreboard Data Extraction From Video", styles["title"]))
story.append(Paragraph(
    "Screenshots of the input, the pipeline running, the scoreboard being detected, "
    "and the extracted output, with a short explanation for each.",
    styles["subtitle"]))
story.append(Spacer(1, 4))

meta = Table(
    [[Paragraph("<b>Input</b><br/>bowling_scoreboard.mp4 &middot; 58s &middot; 1920&times;1080", styles["body"]),
      Paragraph("<b>Ground-truth accuracy</b><br/>34 / 34 cells (frame 1)", styles["body"]),
      Paragraph("<b>Full-video result</b><br/>5 / 6 states clean", styles["body"])]],
    colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch],
)
meta.setStyle(TableStyle([
    ("BOX", (0, 0), (-1, -1), 1, BORDER),
    ("INNERGRID", (0, 0), (-1, -1), 1, BORDER),
    ("BACKGROUND", (0, 0), (-1, -1), WASH),
    ("TOPPADDING", (0, 0), (-1, -1), 10),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
]))
story.append(meta)
story.append(rule())

# ---------- 1. Input frame ----------
story.append(section_header("01", "Input frame"))
story.append(Paragraph(
    "A representative frame sampled from the source video (extracted at 3fps via ffmpeg). "
    "The scoreboard graphic is a fixed on-screen overlay: a lane number, the active player's "
    "name, a 10-frame grid per player, and a team-total row along the bottom.",
    styles["body"]))
story.append(framed_image(SCREENSHOTS / "01_input_frame.png", 6.4 * inch, 3.9 * inch))
story.append(Paragraph("Fig. 1 &mdash; frame_000001.png, the first stable scoreboard frame in the video.",
                        styles["caption"]))
story.append(rule())

# ---------- 2. Code running ----------
story.append(section_header("02", "Code running"))
story.append(Paragraph(
    "The full pipeline runs as a single command once frames are extracted. It samples the "
    "video, discards non-scoreboard frames (the broadcast cuts to bumpers/cartoons between "
    "scoreboard segments), recognizes every cell, cross-checks the result against bowling "
    "scoring rules, and writes structured JSON.",
    styles["body"]))
story.append(framed_image(SCREENSHOTS / "02_code_running.png", 6.4 * inch, 2.6 * inch))
story.append(Paragraph(
    "Fig. 2 &mdash; actual output of <font face='Mono'>python pipeline.py</font>: 174 extracted "
    "frames collapse to 26 stable segments, of which 6 are distinct scorecard states (the rest "
    "are non-scoreboard footage, correctly skipped).",
    styles["caption"]))
story.append(rule())

# ---------- 3. Scoreboard detected ----------
story.append(section_header("03", "Scoreboard detected/extracted"))
story.append(Paragraph(
    "Every cell boundary is calibrated against the graphic's real pixel geometry (measured via "
    "gridline detection, not eyeballed). The green overlay below shows exactly which region is "
    "cropped and fed to the recognizer for every one of the 88 cells in the grid.",
    styles["body"]))
story.append(framed_image(SCREENSHOTS / "03_detected_grid.png", 6.4 * inch, 3.7 * inch))
story.append(Paragraph("Fig. 3 &mdash; the calibrated cell grid overlaid on the input frame.",
                        styles["caption"]))
story.append(Spacer(1, 12))
story.append(Paragraph(
    "The system also detects a transient animation the broadcast pops up after most rolls, "
    "which otherwise corrupts whatever cells it covers. Detection is per row-block against a "
    "known clean-background baseline (not a fixed box), since the animation isn't pinned to one "
    "screen position.",
    styles["body"]))
story.append(framed_image(SCREENSHOTS / "04_overlay_masking.png", 6.4 * inch, 3.7 * inch))
story.append(Paragraph(
    "Fig. 4 &mdash; the overlay correctly detected and masked for the J and V rows in this frame; "
    "those cells are left blank rather than mis-recognized.",
    styles["caption"]))
story.append(rule())

# ---------- 4. Extracted output ----------
story.append(section_header("04", "Extracted output"))
story.append(Paragraph(
    "The final output is one structured JSON entry per distinct scorecard state found in the "
    "video &mdash; every player's per-frame rolls, running totals, lane number, and active player "
    "name. A <font face='Mono'>validation_issues</font> block flags anything a cross-check "
    "against bowling scoring rules found suspect.",
    styles["body"]))
story.append(framed_image(SCREENSHOTS / "05_extracted_output.png", 6.4 * inch, 3.2 * inch))
story.append(Paragraph(
    "Fig. 5 &mdash; excerpt of output/scorecards.json for the frame shown in Fig. 1 (player J shown; "
    "V, P, T follow the same structure). Every value shown matches the source frame exactly.",
    styles["caption"]))

story.append(Spacer(1, 16))
story.append(Paragraph(
    "Full development history &mdash; every problem hit, what changed, and why &mdash; is in "
    "<font face='Mono'>docs/BUILD_LOG.md</font> in the repository.",
    styles["footer"]))

doc = SimpleDocTemplate(
    str(OUT), pagesize=LETTER,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    title="Scoreboard Data Extraction - Documentation",
)
doc.build(story)
print(f"Wrote {OUT}")
