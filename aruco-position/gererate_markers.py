import os

import cv2
import cv2.aruco as aruco
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def generate_aruco_pdf(filename="aruco_markers_targets.pdf", marker_size_cm=2.0, gap_cm=1.0):
    # Wir nutzen das DICT_4X4_50 (reicht für 0-24 IDs völlig aus)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # PDF Setup (A4)
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Einheiten umrechnen
    size = marker_size_cm * cm
    gap = gap_cm * cm

    # Startposition (Oben Links)
    margin_x = 1.5 * cm
    margin_y = 1.5 * cm
    x_pos = margin_x
    y_pos = height - margin_y - size

    print(f"Erstelle PDF: {filename} mit Marker-Größe {marker_size_cm}cm...")

    for marker_id in range(10):
        # Marker Bild generieren (hohe Auflösung für Druck)
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id + 30, 400)
        temp_img = f"temp_id_{marker_id}.png"
        cv2.imwrite(temp_img, marker_img)

        # In PDF zeichnen
        c.drawImage(temp_img, x_pos, y_pos, width=size, height=size)
        c.setFont("Helvetica", 8)
        c.drawString(x_pos, y_pos - 0.3 * cm, f"ID: {marker_id + 30}")

        # Temp Datei löschen
        os.remove(temp_img)

        # Spalten-Logik
        x_pos += size + gap

        # Neue Zeile, wenn rechter Rand erreicht
        if x_pos + size > width - margin_x:
            x_pos = margin_x
            y_pos -= (size + gap + 0.5 * cm)

        # Neue Seite, falls Platz unten ausgeht
        if y_pos < margin_y:
            c.showPage()
            x_pos = margin_x
            y_pos = height - margin_y - size

    c.save()
    print("PDF erfolgreich erstellt!")


if __name__ == "__main__":
    generate_aruco_pdf()
