"""Generate a print-ready PDF of ArUco target markers.

Output: one ArUco marker per A4 page, centred, with the ID printed above
and a footer showing dictionary + physical size. Defaults to the SDC26
target-box marker set:

    31..36   →  target boxes in one team's colour state
    41..46   →  target boxes in the other team's colour state
    99       →  reserved (calibration / spare)

Dictionary: ``DICT_4X4_100`` (the 4×4 family with IDs 0–99, large enough
to cover both colour states without dipping into 5×5+ codes).

Marker size: 18 × 18 cm.  Fits A4 (21 × 29.7 cm) with ~1.5 cm side
margins and ~6 cm above + below for the label and footer.

Run:
    python3 generate_target_markers.py
    python3 generate_target_markers.py --ids 31-36 41-46 99 --size 18 \\
            --out aruco_target_markers.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


DEFAULT_IDS = list(range(31, 37)) + list(range(41, 47)) + [99]
DEFAULT_SIZE_CM = 18.0
DEFAULT_OUT = "aruco_target_markers.pdf"
DEFAULT_DICT = "DICT_4X4_100"


def _parse_ids(tokens: list[str]) -> list[int]:
    """Accept tokens like ``31-36`` (range) or ``99`` (single id)."""
    out: list[int] = []
    for tok in tokens:
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    # de-dup but keep order
    seen: set[int] = set()
    return [i for i in out if not (i in seen or seen.add(i))]


def _dict_from_name(name: str) -> "aruco.Dictionary":
    # cv2.aruco names are uppercase consts on the aruco module
    if not hasattr(aruco, name):
        raise SystemExit(f"unknown dictionary name: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def generate(
    out_path: Path,
    ids: list[int],
    size_cm: float,
    dict_name: str,
    px_per_cm: int = 80,  # 80 px/cm → ~200 DPI on an 18 cm marker
) -> None:
    aruco_dict = _dict_from_name(dict_name)

    page_w, page_h = A4
    size_pt = size_cm * cm
    marker_px = int(size_cm * px_per_cm)

    c = canvas.Canvas(str(out_path), pagesize=A4)

    print(f"Generating: {out_path}")
    print(f"  Dictionary : {dict_name}")
    print(f"  Marker size: {size_cm} × {size_cm} cm  ({marker_px} px @ {px_per_cm} px/cm)")
    print(f"  IDs        : {ids}")
    print(f"  Pages      : {len(ids)}")

    tmp_dir = out_path.parent
    for marker_id in ids:
        img = aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        tmp_png = tmp_dir / f"_target_marker_{marker_id}.png"
        cv2.imwrite(str(tmp_png), img)

        # Centre the marker; nudge down so the label has breathing room.
        x = (page_w - size_pt) / 2
        y = (page_h - size_pt) / 2 - 1.0 * cm

        c.drawImage(str(tmp_png), x, y, width=size_pt, height=size_pt)

        # Big label above the marker.
        c.setFont("Helvetica-Bold", 30)
        label = f"ID: {marker_id}"
        label_w = c.stringWidth(label, "Helvetica-Bold", 30)
        c.drawString((page_w - label_w) / 2, y + size_pt + 1.2 * cm, label)

        # Footer with dict + size.
        c.setFont("Helvetica", 10)
        footer = f"{dict_name}  |  {size_cm:g} × {size_cm:g} cm  |  ID {marker_id}"
        footer_w = c.stringWidth(footer, "Helvetica", 10)
        c.drawString((page_w - footer_w) / 2, y - 1.0 * cm, footer)

        # Cut marks at the corners of the marker rectangle (3 mm lines)
        c.setLineWidth(0.4)
        cut = 0.3 * cm
        for cx, cy in [(x, y), (x + size_pt, y),
                       (x, y + size_pt), (x + size_pt, y + size_pt)]:
            c.line(cx - cut, cy, cx + cut, cy)
            c.line(cx, cy - cut, cx, cy + cut)

        tmp_png.unlink(missing_ok=True)
        c.showPage()

    c.save()
    print(f"Wrote {out_path}  ({len(ids)} pages)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name(DEFAULT_OUT),
                   help=f"output PDF path (default: {DEFAULT_OUT} next to this script)")
    p.add_argument("--ids", nargs="+", default=None,
                   help="marker IDs to emit. Accepts single (99) or ranges (31-36). "
                        "Default: 31-36 41-46 99.")
    p.add_argument("--size", type=float, default=DEFAULT_SIZE_CM,
                   help=f"marker side length in cm (default: {DEFAULT_SIZE_CM})")
    p.add_argument("--dict", dest="dict_name", default=DEFAULT_DICT,
                   help=f"ArUco dictionary name (default: {DEFAULT_DICT})")
    p.add_argument("--px-per-cm", type=int, default=80,
                   help="raster density of the embedded PNG (default 80 ≈ 200 DPI)")
    args = p.parse_args()

    ids = _parse_ids(args.ids) if args.ids else DEFAULT_IDS

    # Sanity: marker fits A4 with margin
    if args.size * cm + 4 * cm > A4[1]:
        print(f"warning: {args.size} cm marker leaves <2 cm vertical margin on A4",
              file=sys.stderr)

    generate(args.out, ids, args.size, args.dict_name, args.px_per_cm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
