"""
Generate a PDF with ArUco markers (DICT_4X4_100), one marker per A4 page.
Each marker is 18x18 cm, centered on the page, with the marker ID printed above it.
Contains marker IDs 1-24 and 61-76 (40 pages total).
"""

import os

import cv2
import cv2.aruco as aruco
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def generate_aruco_pdf(filename="aruco_markers_4x4_100.pdf", marker_size_cm=10.0):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

    c = canvas.Canvas(filename, pagesize=A4)
    page_w, page_h = A4
    size = marker_size_cm * cm

    # IDs: 1-24 (reference/pillar markers) and 61-76 (target markers)
    marker_ids = list(range(7, 11)) + list(range(0, 1))

    print(f"Generating PDF: {filename}")
    print(f"  Dictionary: DICT_4X4_100")
    print(f"  Marker size: {marker_size_cm} cm")
    print(f"  Marker IDs: 1-24, 61-76 ({len(marker_ids)} pages)")

    for marker_id in marker_ids:
        # Generate marker image at high resolution for print quality
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, 1200)
        temp_img = f"_temp_marker_{marker_id}.png"
        cv2.imwrite(temp_img, marker_img)

        # Center marker on page
        x = (page_w - size) / 2
        y = (page_h - size) / 2 - 1.0 * cm  # shift down slightly to make room for label

        # Draw marker
        c.drawImage(temp_img, x, y, width=size, height=size)

        # Label above marker
        c.setFont("Helvetica", 24)
        label = f"ArUco 4x4_100  —  ID: {marker_id}"
        label_w = c.stringWidth(label, "Helvetica", 24)
        c.drawString((page_w - label_w) / 2, y + size + 1.0 * cm, label)

        # Small footer with size info
        c.setFont("Helvetica", 10)
        footer = f"DICT_4X4_100 | {marker_size_cm}x{marker_size_cm} cm | ID {marker_id}"
        footer_w = c.stringWidth(footer, "Helvetica", 10)
        c.drawString((page_w - footer_w) / 2, y - 1.0 * cm, footer)

        os.remove(temp_img)

        c.showPage()

    c.save()
    print(f"Done! {len(marker_ids)} pages written to {filename}")


if __name__ == "__main__":
    generate_aruco_pdf()
