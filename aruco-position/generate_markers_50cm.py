"""
Generate a PDF with ArUco markers (DICT_4X4_1000), one marker per page,
on a square 50 x 50 cm page (custom size, NOT a DIN format). The marker
fills the page edge to edge by default. Default: 16 pages, IDs 1..16.

Usage:
    python generate_markers_50cm.py
    python generate_markers_50cm.py --first 1 --last 16 --size-cm 50 \
        --margin-cm 0 --output aruco_4x4_1000_50cm_ids_1-16.pdf

Notes:
    * The page IS the marker. With --margin-cm 0 there is no whitespace
      and no label on the page; the file is just sixteen 50 x 50 cm
      ArUco squares.
    * If you want a small white border (for cutting tolerance / safe
      handling) pass e.g. --margin-cm 1.0; the marker shrinks to fit
      and the printable area still measures 50 x 50 cm.
    * Pass --label to print a tiny ID label in the bottom margin
      (requires --margin-cm >= 1.0 so the label fits).

Requires: opencv-contrib-python>=4.7  reportlab
"""

import argparse
import os
import sys

import cv2
import cv2.aruco as aruco
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


# Pixels per centimetre when rasterising the marker. ArUco markers are
# pure black/white squares so a moderate raster (~47 px/cm = ~120 dpi
# at 1:1) prints crisply; raising it just bloats the PDF.
PX_PER_CM = 47


def generate(
    output: str,
    first_id: int,
    last_id: int,
    size_cm: float,
    margin_cm: float,
    show_label: bool,
    dict_name: str = "DICT_4X4_1000",
) -> None:
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

    page_w = page_h = size_cm * cm
    marker_side = (size_cm - 2 * margin_cm) * cm
    if marker_side <= 0:
        sys.exit("error: --margin-cm too large; marker would have zero size")
    if show_label and margin_cm < 1.0:
        sys.exit("error: --label needs --margin-cm >= 1.0 to fit the text")

    marker_ids = list(range(first_id, last_id + 1))
    px = max(800, int((size_cm - 2 * margin_cm) * PX_PER_CM))

    c = canvas.Canvas(output, pagesize=(page_w, page_h))

    print(f"Generating {output}")
    print(f"  Dictionary : {dict_name}")
    print(f"  Page       : {size_cm} x {size_cm} cm (square, custom)")
    print(f"  Margin     : {margin_cm} cm")
    print(f"  Marker     : {(size_cm - 2 * margin_cm):.1f} x "
          f"{(size_cm - 2 * margin_cm):.1f} cm  ({px} x {px} px raster)")
    print(f"  IDs        : {first_id}..{last_id} ({len(marker_ids)} pages)")
    if show_label:
        print(f"  Label      : on (in bottom margin)")

    for marker_id in marker_ids:
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, px)
        temp_img = f"_tmp_marker_{marker_id}.png"
        cv2.imwrite(temp_img, marker_img)

        x = margin_cm * cm
        y = margin_cm * cm
        c.drawImage(temp_img, x, y, width=marker_side, height=marker_side)

        if show_label:
            c.setFont("Helvetica", 10)
            footer = (
                f"{dict_name} | "
                f"{(size_cm - 2 * margin_cm):.1f}x{(size_cm - 2 * margin_cm):.1f} cm | "
                f"ID {marker_id} | print 100%"
            )
            footer_w = c.stringWidth(footer, "Helvetica", 10)
            c.drawString((page_w - footer_w) / 2, 0.3 * cm, footer)

        os.remove(temp_img)
        c.showPage()

    c.save()
    print(f"Done. {len(marker_ids)} pages written to {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=int, default=1, help="first marker ID (default 1)")
    p.add_argument("--last", type=int, default=16, help="last marker ID (default 16)")
    p.add_argument("--size-cm", type=float, default=50.0,
                   help="page side length in cm; the page is square (default 50)")
    p.add_argument("--margin-cm", type=float, default=0.0,
                   help="white border around the marker in cm (default 0 -> "
                        "marker fills the page)")
    p.add_argument("--label", action="store_true",
                   help="print a small ID label in the bottom margin "
                        "(needs --margin-cm >= 1.0)")
    p.add_argument("--dict", dest="dict_name", default="DICT_4X4_1000",
                   help="ArUco dictionary (default DICT_4X4_1000)")
    p.add_argument("--output", "-o",
                   default="aruco_4x4_1000_50cm_ids_1-16.pdf",
                   help="output PDF path")
    args = p.parse_args()

    if args.first > args.last:
        sys.exit("error: --first must be <= --last")

    generate(
        output=args.output,
        first_id=args.first,
        last_id=args.last,
        size_cm=args.size_cm,
        margin_cm=args.margin_cm,
        show_label=args.label,
        dict_name=args.dict_name,
    )


if __name__ == "__main__":
    main()
