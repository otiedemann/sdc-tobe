import cv2
import numpy as np
import os
import time

INPUT_DIR  = r"C:\Users\Dell\Documents\Python_MER\Drohne\ArucoMarker"
IMAGE_NAME = "aruco_scene.jpeg"
image_path = os.path.join(INPUT_DIR, IMAGE_NAME)


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _make_params(win_min: int, win_max: int, win_step: int,
                 min_perim: float, poly_acc: float, error_rate: float,
                 refine=cv2.aruco.CORNER_REFINE_NONE   # ← NONE: schnellste Variante
                 ) -> cv2.aruco.DetectorParameters:
    """
    Erzeugt DetectorParameters – speed-optimiert.

    Geschwindigkeits-Hebel im Überblick:
      win_step (größter Hebel) : Anzahl Schwellwert-Stufen = (max-min)/step
                                  step=1  → viele Stufen, langsam
                                  step=10 → wenige Stufen, schnell
      win_max (zweitgrößter)   : Größere Fenster = mehr Rechenaufwand
                                  31 ist ein guter Kompromiss für Drohnenbilder
      perspectiveRemovePixelPerCell : 4 statt 8 → halb so viele Pixel beim Bit-Lesen
      cornerRefinementMethod   : NONE < CONTOUR < SUBPIX < APRILTAG (Kosten)
    """
    p = cv2.aruco.DetectorParameters()

    # ── Adaptive Schwellwertfenster ───────────────────────────────────────────
    p.adaptiveThreshWinSizeMin  = win_min
    p.adaptiveThreshWinSizeMax  = win_max   # 31: ausreichend für mittelgroße Marker
    p.adaptiveThreshWinSizeStep = win_step  # GROSSER HEBEL: höher = schneller

    # ── Marker-Größenfilter ───────────────────────────────────────────────────
    p.minMarkerPerimeterRate    = min_perim
    p.maxMarkerPerimeterRate    = 4.0       # groß genug für alle Drohnen-Distanzen

    # ── Viereck-Erkennung ─────────────────────────────────────────────────────
    p.polygonalApproxAccuracyRate = poly_acc
    p.minCornerDistanceRate       = 0.02
    p.minDistanceToBorder         = 1

    # ── Bit-Dekodierung ───────────────────────────────────────────────────────
    p.errorCorrectionRate              = error_rate
    p.perspectiveRemovePixelPerCell    = 4     # ← 4 statt 8: schneller, kaum Verlust
    p.perspectiveRemoveIgnoredMarginPerCell = 0.13

    # ── Ecken-Verfeinerung ────────────────────────────────────────────────────
    p.cornerRefinementMethod = refine      # NONE = kein Subpixel-Schritt → schnellste

    p.markerBorderBits       = 1
    p.minMarkerDistanceRate  = 0.02

    return p


def _confidence(c: np.ndarray) -> float:
    """
    Schnelle Confidence-Berechnung für ein (4,2)-Ecken-Array.
    Konvexität analytisch – kein cv2.convexHull nötig.
    """
    edges = [
        np.linalg.norm(c[0] - c[1]),
        np.linalg.norm(c[1] - c[2]),
        np.linalg.norm(c[2] - c[3]),
        np.linalg.norm(c[3] - c[0]),
    ]
    max_e = max(edges)
    squareness = min(edges) / max_e if max_e > 0 else 0.0

    d1 = c[2] - c[0]
    d2 = c[3] - c[1]
    area_marker = abs(d1[0] * d2[1] - d1[1] * d2[0]) / 2.0

    def tri(a, b, cc): return abs((b[0]-a[0])*(cc[1]-a[1]) - (cc[0]-a[0])*(b[1]-a[1])) / 2.0
    area_hull = max(
        tri(c[0], c[1], c[2]) + tri(c[0], c[2], c[3]),
        tri(c[0], c[1], c[3]) + tri(c[1], c[2], c[3]),
    )
    convexity = area_marker / area_hull if area_hull > 0 else 0.0

    return round(0.6 * squareness + 0.4 * convexity, 4)


# ─── Pässe ────────────────────────────────────────────────────────────────────
#
#  Modi im Überblick:
#
#  ultrafast → 1 Pass,  REFINE_NONE,     step=10  → maximale Geschwindigkeit
#  fast      → 1 Pass,  REFINE_NONE,     step=6   → gut für klare Drohnenbilder
#  balanced  → 2 Pässe, REFINE_CONTOUR,  step=6   → Fallback bei schrägen Markern
#  full      → 4 Pässe, REFINE_CONTOUR,  step=4   → maximale Erkennungsrate
#
#  Faustregel: ultrafast/fast für Echtzeit, balanced für normale Aufnahmen,
#              full wenn Marker regelmäßig nicht erkannt werden.

_PASSES_ULTRAFAST: list[tuple[str, cv2.aruco.DetectorParameters]] = [
    # 1 Pass, step=10 → (31-3)/10 = ~3 Schwellwert-Stufen, REFINE_NONE
    ("gray", _make_params(3, 31, 10, 0.010, 0.05, 0.6,
                          refine=cv2.aruco.CORNER_REFINE_NONE)),
]

_PASSES_FAST: list[tuple[str, cv2.aruco.DetectorParameters]] = [
    # 1 Pass, step=6 → (31-3)/6 = ~5 Schwellwert-Stufen, REFINE_NONE
    ("gray", _make_params(3, 31, 6,  0.008, 0.05, 0.6,
                          refine=cv2.aruco.CORNER_REFINE_NONE)),
]

_PASSES_BALANCED: list[tuple[str, cv2.aruco.DetectorParameters]] = [
    # Pass 1: Standard
    ("gray",  _make_params(3, 31, 6,  0.008, 0.05, 0.6,
                           refine=cv2.aruco.CORNER_REFINE_CONTOUR)),
    # Pass 2: CLAHE für schräge / schattierte Marker
    ("clahe", _make_params(3, 31, 6,  0.005, 0.08, 0.8,
                           refine=cv2.aruco.CORNER_REFINE_CONTOUR)),
]

_PASSES_FULL: list[tuple[str, cv2.aruco.DetectorParameters]] = [
    # Pass 1: Standard
    ("gray",    _make_params(3, 31, 4,  0.008, 0.05, 0.6,
                             refine=cv2.aruco.CORNER_REFINE_CONTOUR)),
    # Pass 2: CLAHE – Schatten / ungleichmäßige Beleuchtung
    ("clahe",   _make_params(3, 31, 4,  0.005, 0.08, 0.8,
                             refine=cv2.aruco.CORNER_REFINE_CONTOUR)),
    # Pass 3: Deblur – Bewegungsunschärfe
    ("deblur",  _make_params(3, 27, 4,  0.006, 0.08, 0.8,
                             refine=cv2.aruco.CORNER_REFINE_CONTOUR)),
    # Pass 4: Upscale ×2 – kleine / weit entfernte Marker
    ("upscale", _make_params(3, 21, 4,  0.010, 0.06, 0.7,
                             refine=cv2.aruco.CORNER_REFINE_CONTOUR)),
]

# Modus-Map für detect_aruco()
_MODE_MAP = {
    "ultrafast": _PASSES_ULTRAFAST,
    "fast":      _PASSES_FAST,
    "balanced":  _PASSES_BALANCED,
    "full":      _PASSES_FULL,
}


# ─── Erkennung ────────────────────────────────────────────────────────────────

def detect_aruco(image_path: str,
                 aruco_dict_type=cv2.aruco.DICT_4X4_50,
                 min_confidence: float = 0.3,
                 mode: str = "fast") -> tuple[list[dict], dict]:
    """
    Erkennt ArUco-Marker in einem Bild.

    Koordinatensystem:
        Ursprung = links unten, X nach rechts, Y nach oben.

    Winkel:
        0° = Marker-X zeigt rechts, +90° = oben. Bereich (-180°, +180°].

    Confidence (0.0–1.0):
        0.6 × Quadratizität + 0.4 × Konvexität. 1.0 = idealer Marker.

    Modi (mode=):
        "ultrafast" → 1 Pass, step=10, REFINE_NONE   (~maximale Geschwindigkeit)
        "fast"      → 1 Pass, step=6,  REFINE_NONE   (Standard, gute Balance)
        "balanced"  → 2 Pässe, REFINE_CONTOUR         (schräge/schattierte Marker)
        "full"      → 4 Pässe, REFINE_CONTOUR         (maximale Erkennungsrate)

    Rückgabe:
        (markers_sorted_by_id, corners_map {id: (4,2)})
    """
    if mode not in _MODE_MAP:
        raise ValueError(f"Unbekannter Modus '{mode}'. Erlaubt: {list(_MODE_MAP)}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")

    h, w = image.shape[:2]
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    # ── Vorverarbeitung (lazy) ────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    prep: dict[str, np.ndarray | None] = {
        "gray":    gray,
        "clahe":   None,
        "deblur":  None,
        "upscale": None,
    }

    def get_prep(key: str) -> np.ndarray:
        if prep[key] is None:
            if key == "clahe":
                prep[key] = cv2.createCLAHE(clipLimit=3.0,
                                             tileGridSize=(8, 8)).apply(gray)
            elif key == "deblur":
                blur_tmp   = cv2.GaussianBlur(gray, (0, 0), 2.0)
                prep[key]  = cv2.addWeighted(gray, 1.6, blur_tmp, -0.6, 0)
            elif key == "upscale":
                prep[key]  = cv2.resize(gray, None, fx=2.0, fy=2.0,
                                        interpolation=cv2.INTER_LINEAR)
        return prep[key]

    # ── Detektoren aus gewähltem Modus ────────────────────────────────────────
    active_passes = _MODE_MAP[mode]
    detectors = [
        (prep_key, cv2.aruco.ArucoDetector(aruco_dict, params))
        for prep_key, params in active_passes
    ]

    CONF_PERFECT = 0.97

    found:       dict[int, dict]       = {}
    corners_map: dict[int, np.ndarray] = {}

    for prep_key, detector in detectors:
        if found and all(m["confidence"] >= CONF_PERFECT for m in found.values()):
            break

        # Upscale: Koordinaten nach Detektion ×0.5 zurückskalieren
        scale = 0.5 if prep_key == "upscale" else 1.0
        src   = get_prep(prep_key)

        corners, ids, _ = detector.detectMarkers(src)
        if ids is None:
            continue

        for i, marker_id in enumerate(ids.flat):
            mid = int(marker_id)

            if mid in found and found[mid]["confidence"] >= CONF_PERFECT:
                continue

            c    = corners[i][0] * scale   # (4,2): TL TR BR BL
            conf = _confidence(c)

            if conf < min_confidence:
                continue
            if mid in found and conf <= found[mid]["confidence"]:
                continue

            cx, cy_img = np.mean(c, axis=0)
            x = float(cx)
            y = float(h - cy_img)

            mid_left  = (c[0] + c[3]) * 0.5
            mid_right = (c[1] + c[2]) * 0.5
            dx =  float(mid_right[0] - mid_left[0])
            dy = -float(mid_right[1] - mid_left[1])
            angle = float(np.degrees(np.arctan2(dy, dx)))

            d1 = c[2] - c[0]
            d2 = c[3] - c[1]
            area_px2 = abs(float(d1[0]*d2[1] - d1[1]*d2[0])) / 2.0

            found[mid] = {
                "id":         mid,
                "x":          round(x, 1),
                "y":          round(y, 1),
                "angle_deg":  round(angle, 2),
                "confidence": conf,
                "area_px2":   round(area_px2, 1),
            }
            corners_map[mid] = c.copy()

    # ── Filtern & sortieren ───────────────────────────────────────────────────
    results = sorted(
        (m for m in found.values() if m["confidence"] >= min_confidence),
        key=lambda m: m["id"],
    )
    valid_ids   = {m["id"] for m in results}
    corners_map = {k: v for k, v in corners_map.items() if k in valid_ids}

    return results, corners_map


# ─── Visualisierung ───────────────────────────────────────────────────────────

def visualize_aruco(image_path: str, markers: list[dict],
                    corners_map: dict[int, np.ndarray],
                    save_path: str | None = None,
                    show: bool = True) -> np.ndarray:
    """
    Zeichnet erkannte ArUco-Marker ins Originalbild:
      - Farbiger Rahmen (Confidence: grün→gelb→rot)
      - Ecken nummeriert (0=TL, 1=TR, 2=BR, 3=BL)
      - Pfeil für Marker-X-Achse (Winkelrichtung)
      - Label: ID, Winkel, Confidence
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")

    h, w = image.shape[:2]
    font_scale = max(0.5, w / 2500)
    thickness  = max(1, int(w / 900))
    font       = cv2.FONT_HERSHEY_SIMPLEX

    for m in markers:
        mid  = m["id"]
        conf = m["confidence"]
        c    = corners_map[mid]

        color = (0, 220, 0) if conf >= 0.9 else (0, 200, 220) if conf >= 0.8 else (0, 80, 220)

        cv2.polylines(image, [c.astype(np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=color, thickness=thickness * 2)

        for idx, (px, py) in enumerate(c.astype(int)):
            cv2.circle(image, (px, py), thickness * 4, color, -1)
            cv2.putText(image, ["0:TL", "1:TR", "2:BR", "3:BL"][idx],
                        (px + 6, py - 6), font, font_scale * 0.7,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        center_img = np.mean(c, axis=0).astype(int)
        cv2.circle(image, tuple(center_img), thickness * 3, color, -1)

        angle_rad = np.radians(m["angle_deg"])
        arrow_len = int(np.sqrt(m["area_px2"]) * 0.45)
        arrow_end = (
            int(center_img[0] + arrow_len * np.cos(angle_rad)),
            int(center_img[1] - arrow_len * np.sin(angle_rad)),
        )
        cv2.arrowedLine(image, tuple(center_img), arrow_end,
                        (255, 80, 0), thickness * 2, tipLength=0.25)

        label_lines = [f"ID {mid}", f"{m['angle_deg']:+.1f}°", f"conf {conf:.2f}"]
        line_h = int(22 * font_scale * 1.6)
        lx = center_img[0] + 12
        ly = center_img[1] - line_h * (len(label_lines) - 1) // 2
        for j, line in enumerate(label_lines):
            pos = (lx, ly + j * line_h)
            cv2.putText(image, line, (pos[0]+2, pos[1]+2), font,
                        font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(image, line, pos, font,
                        font_scale, color, thickness, cv2.LINE_AA)

    legend = [
        f"Erkannte Marker: {len(markers)}",
        "Farbe: gruen>=0.9  gelb>=0.8  rot>=0.5",
        "Pfeil: Marker-X-Achse (Winkel)",
    ]
    for j, line in enumerate(legend):
        ly = 30 + j * int(28 * font_scale)
        cv2.putText(image, line, (12, ly + 2), font,
                    font_scale * 0.85, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(image, line, (10, ly), font,
                    font_scale * 0.85, (255, 255, 255), thickness, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Annotiertes Bild gespeichert: {save_path}")

    if show:
        cv2.namedWindow("ArUco Erkennung", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ArUco Erkennung", min(w, 1400), min(h, 900))
        cv2.imshow("ArUco Erkennung", image)
        print("Fenster geöffnet – beliebige Taste drücken zum Schließen.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


# ─── Hauptprogramm ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Modus wählen:
    #   "ultrafast" → Echtzeit / maximale Geschwindigkeit
    #   "fast"      → Standard (1 Pass, gute Balance)
    #   "balanced"  → schräge / schattierte Marker (2 Pässe)
    #   "full"      → maximale Erkennungsrate (4 Pässe)
    MODE = "ultrafast"

    t_start = time.perf_counter()
    markers, corners_map = detect_aruco(image_path, min_confidence=0.5, mode=MODE)
    t_stop  = time.perf_counter()
    print(f"Erkennungsdauer: {t_stop - t_start:.4f} s  (mode='{MODE}')")

    if not markers:
        print("Keine ArUco-Marker erkannt.")
        naechster = {"ultrafast": "fast", "fast": "balanced",
                     "balanced": "full", "full": "full"}
        print(f"Tipp: mode='{naechster[MODE]}' versuchen.")
    else:
        print(f"Bild:                {image_path}")
        print(f"Erkannte Marker:     {len(markers)}")
        print(f"Koordinatenursprung: links unten  |  Y-Achse: nach oben positiv\n")
        print(f"{'ID':>4}  {'X (px)':>9}  {'Y (px)':>9}  {'Winkel (°)':>11}  "
              f"{'Confidence':>11}  {'Fläche (px²)':>13}")
        print("─" * 68)
        for m in markers:
            print(f"{m['id']:>4}  {m['x']:>9.1f}  {m['y']:>9.1f}  "
                  f"{m['angle_deg']:>11.2f}  {m['confidence']:>11.4f}  "
                  f"{m['area_px2']:>13.1f}")

        base, ext = os.path.splitext(image_path)
        visualize_aruco(image_path, markers, corners_map,
                        save_path=base + "_annotated" + ext, show=True)