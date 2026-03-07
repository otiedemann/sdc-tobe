"""
generate_aruco_scene.py
───────────────────────
Generiert ein synthetisches Bild mit zufällig verteilten ArUco-Markern
(DICT_4X4_50) in verschiedenen Winkeln, Abständen und Rauschstufen.

Auflösung: 1920×1080 (Full HD)
Ausgabe:   aruco_scene.jpeg
"""

import cv2
import numpy as np
import random
import os

# ─── Konfiguration ────────────────────────────────────────────────────────────

W, H        = 1280,720
JPEG_Q      = 80            # JPEG-Qualität (1–100, kleiner = stärker komprimiert)

RNG_SEED    = None          # None = zufällig; int = reproduzierbar
N_MARKERS   = 20            # None = zufällig 3–8; int = feste Anzahl

OUTPUT_DIR  = r"C:\Users\Dell\Documents\Python_MER\Drohne\ArucoMarker"
IMAGE_NAME   = "aruco_scene.jpeg"

# Rauscharten (True = aktiviert)
NOISE_CONFIG = {
    "gaussian":      True,   # Gaußsches Bildrauschen
    "salt_pepper":   True,   # Salz-und-Pfeffer-Rauschen
    "blur_local":    True,   # Lokale Unschärfe auf einzelnen Markern
    "illumination":  True,   # Ungleichmäßige Beleuchtung / Vignettierung
    "jpeg_artifact": True,   # JPEG-Kompressionsartefakte (Zwischenspeicherung)
    "perspective":   True,   # Perspektivische Verzerrung der Marker
    "occlusion":     True,   # Zufällige Verdeckung (dunkle Rechtecke)
}

# ─── RNG initialisieren ───────────────────────────────────────────────────────

seed = RNG_SEED if RNG_SEED is not None else random.randint(0, 99999) 
print(f"Seed: {seed}")
rng  = np.random.RandomState(seed)
prng = random.Random(seed)

# ─── Hintergrund: neutrale Wand ───────────────────────────────────────────────

def make_background(w, h, rng):
    base_color = rng.randint(160, 200)
    bg = np.full((h, w, 3), base_color, dtype=np.uint8)
    for _ in range(600):
        x1 = rng.randint(0, w)
        y1 = rng.randint(0, h)
        x2 = x1 + rng.randint(-300, 300)
        y2 = y1 + rng.randint(-5, 5)
        shade = int(np.clip(int(base_color) + int(rng.randint(-12, 12)), 0, 255))
        cv2.line(bg, (int(x1), int(y1)), (int(x2), int(y2)), (shade, shade, shade), 1)
    return bg

# ─── ArUco Marker generieren ──────────────────────────────────────────────────

def get_marker(marker_id: int, size: int) -> np.ndarray:
    """Gibt einen ArUco-Marker als BGR-Bild zurück."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_gray = np.zeros((size, size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, size, marker_gray, 1)
    return cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)


def apply_perspective(marker: np.ndarray, tilt_x: float, tilt_y: float) -> np.ndarray:
    """Simuliert perspektivische Neigung (tilt 0.0=frontal, 0.45=stark)."""
    s  = marker.shape[0]
    tx = tilt_x * s * 0.4
    ty = tilt_y * s * 0.4
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    dst = np.float32([
        [ tx,      ty      ],
        [ s - tx, -ty      ],
        [ s + tx,  s + ty  ],
        [-tx,      s - ty  ],
    ])
    min_x = dst[:, 0].min(); min_y = dst[:, 1].min()
    dst[:, 0] -= min_x;      dst[:, 1] -= min_y
    out_w = int(dst[:, 0].max()) + 1
    out_h = int(dst[:, 1].max()) + 1
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(marker, M, (out_w, out_h),
                               flags=cv2.INTER_LINEAR,
                               borderValue=(128, 128, 128))


def rotate_marker(marker: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotiert das Marker-Bild um angle_deg Grad."""
    h, w   = marker.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(marker, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderValue=(128, 128, 128))


def add_local_blur(marker: np.ndarray, rng, strength: int) -> np.ndarray:
    """Weichzeichner: simuliert Bewegungsunschärfe / Defokus."""
    if strength <= 0:
        return marker
    k = strength * 2 + 1
    return cv2.GaussianBlur(marker, (k, k), strength * 0.5)


def add_local_noise(marker: np.ndarray, rng, sigma: float) -> np.ndarray:
    """Gaußsches Rauschen auf einzelnem Marker."""
    noise = rng.normal(0, sigma, marker.shape).astype(np.int16)
    return np.clip(marker.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# ─── Marker auf Canvas platzieren ─────────────────────────────────────────────

def paste_marker(canvas: np.ndarray, marker: np.ndarray,
                 cx: int, cy: int, alpha: float = 1.0):
    """Klebt marker zentriert auf (cx, cy); blendet grauen Rotationsrand aus."""
    mh, mw = marker.shape[:2]
    x0 = cx - mw // 2;  y0 = cy - mh // 2
    x1 = x0 + mw;       y1 = y0 + mh
    cx0 = max(0, x0);   cy0 = max(0, y0)
    cx1 = min(canvas.shape[1], x1); cy1 = min(canvas.shape[0], y1)
    mx0 = cx0 - x0; my0 = cy0 - y0
    mx1 = mx0 + cx1 - cx0; my1 = my0 + cy1 - cy0
    if cx1 <= cx0 or cy1 <= cy0:
        return
    region = marker[my0:my1, mx0:mx1]
    gray_mask = (
        (region[:, :, 0] > 100) & (region[:, :, 0] < 155) &
        (region[:, :, 1] > 100) & (region[:, :, 1] < 155) &
        (region[:, :, 2] > 100) & (region[:, :, 2] < 155)
    )
    mask = (~gray_mask).astype(np.float32)[:, :, np.newaxis] * alpha
    canvas[cy0:cy1, cx0:cx1] = (
        region * mask + canvas[cy0:cy1, cx0:cx1] * (1 - mask)
    ).astype(np.uint8)

# ─── Globale Rauscheffekte ────────────────────────────────────────────────────

def add_gaussian_noise(image: np.ndarray, rng, sigma: float) -> np.ndarray:
    noise = rng.normal(0, sigma, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(image: np.ndarray, rng, density: float) -> np.ndarray:
    out = image.copy()
    n   = int(density * image.size / 3)
    xs  = rng.randint(0, image.shape[1], n)
    ys  = rng.randint(0, image.shape[0], n)
    out[ys[:n//2], xs[:n//2]] = 255
    out[ys[n//2:], xs[n//2:]] = 0
    return out


def add_vignette(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    h, w   = image.shape[:2]
    Y, X   = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vig    = np.clip(1.0 - strength * dist, 0, 1).astype(np.float32)
    return np.clip(image.astype(np.float32) * vig[:, :, np.newaxis],
                   0, 255).astype(np.uint8)


def add_illumination_gradient(image: np.ndarray, rng) -> np.ndarray:
    h, w  = image.shape[:2]
    lx    = rng.uniform(0.1, 0.9) * w
    ly    = rng.uniform(0.0, 0.5) * h
    Y, X  = np.ogrid[:h, :w]
    dist  = np.sqrt((X - lx) ** 2 + (Y - ly) ** 2)
    max_d = np.sqrt(w ** 2 + h ** 2)
    grad  = np.clip(1.0 - 0.4 * dist / max_d, 0.6, 1.0).astype(np.float32)
    return np.clip(image.astype(np.float32) * grad[:, :, np.newaxis],
                   0, 255).astype(np.uint8)


def add_jpeg_artifacts(image: np.ndarray, quality: int = 25) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_occlusion(image: np.ndarray, rng, n: int = 3) -> np.ndarray:
    out  = image.copy()
    h, w = image.shape[:2]
    for _ in range(n):
        x     = rng.randint(0, w - 80)
        y     = rng.randint(0, h - 80)
        bw    = rng.randint(30, 120)
        bh    = rng.randint(20, 80)
        color = int(rng.randint(0, 60))
        cv2.rectangle(out, (int(x), int(y)),
                      (int(x + bw), int(y + bh)), (color, color, color), -1)
    return out

# ─── Hauptgenerator ───────────────────────────────────────────────────────────

def generate_scene(n_markers=None):
    if n_markers is None:
        n_markers = prng.randint(3, 8)

    # DICT_4X4_50 hat 50 IDs (0–49)
    n_markers  = min(n_markers, 50)
    marker_ids = prng.sample(range(50), n_markers)

    print(f"Generiere {n_markers} Marker …")
    canvas       = make_background(W, H, rng)
    ground_truth = []
    placed       = []

    for marker_id in marker_ids:

        # Größe: 60–200px (unter 60px kaum erkennbar)
        size_base  = prng.randint(20, 180)
        angle_deg  = prng.uniform(-10, 10)
        tilt_x     = prng.uniform(-0.4, 0.4) if NOISE_CONFIG["perspective"] else 0.0
        tilt_y     = prng.uniform(-0.4, 0.4) if NOISE_CONFIG["perspective"] else 0.0
        blur_str   = prng.randint(0, 2)         if NOISE_CONFIG["blur_local"]  else 0
        noise_sig  = prng.uniform(5, 20)        if NOISE_CONFIG["gaussian"]    else 0.0
        brightness = prng.uniform(0.8, 1.2)

        # Marker aufbauen
        m = get_marker(marker_id, size_base)
        if NOISE_CONFIG["perspective"]:
            m = apply_perspective(m, tilt_x, tilt_y)
        m = rotate_marker(m, angle_deg)
        if NOISE_CONFIG["blur_local"] and blur_str > 0:
            m = add_local_blur(m, rng, blur_str)
        if NOISE_CONFIG["gaussian"] and noise_sig > 0:
            m = add_local_noise(m, rng, noise_sig)
        m = np.clip(m.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        mh, mw = m.shape[:2]
        GAP = 10   # Mindestabstand in Pixeln

        # Kollisionsfreie Position suchen (AABB mit Mindestabstand)
        # placed speichert (cx, cy, half_w, half_h) jedes platzierten Markers
        hw = mw // 2
        hh = mh // 2

        placed_ok = False
        for _ in range(500):
            cx = prng.randint(hw, W - hw)
            cy = prng.randint(hh, H - hh)
            ok = True
            for (px, py, phw, phh) in placed:
                # AABB-Überlappung + GAP: kein Überlapp wenn Abstand in x ODER y > Summe der Hälften + GAP
                if (abs(cx - px) < hw + phw + GAP and
                        abs(cy - py) < hh + phh + GAP):
                    ok = False
                    break
            if ok:
                placed_ok = True
                break

        if not placed_ok:
            print(f"  Warnung: Kein kollisionsfreier Platz für ID {marker_id} – übersprungen.")
            continue

        paste_marker(canvas, m, cx, cy)
        placed.append((cx, cy, hw, hh))

        # Ground-Truth (Ursprung links unten)
        x_gt = cx
        y_gt = H - cy
        ground_truth.append({
            "id":          marker_id,
            "x":           x_gt,
            "y":           y_gt,
            "angle_deg":   round(angle_deg, 2),
            "size_px":     size_base,
            "tilt_x":      round(tilt_x, 3),
            "tilt_y":      round(tilt_y, 3),
            "blur":        blur_str,
            "noise_sigma": round(noise_sig, 2),
            "brightness":  round(brightness, 3),
        })
        print(f"  ID {marker_id:>2d} | pos ({x_gt:4d},{y_gt:4d}) | "
              f"Winkel {angle_deg:+6.1f}° | Größe {size_base}px | "
              f"Blur {blur_str} | σ={noise_sig:.1f}")

    # Globale Effekte
    if NOISE_CONFIG["illumination"]:
        canvas = add_illumination_gradient(canvas, rng)
        canvas = add_vignette(canvas, strength=prng.uniform(0.25, 0.55))
    if NOISE_CONFIG["salt_pepper"]:
        canvas = add_salt_pepper(canvas, rng, density=prng.uniform(0.001, 0.004))
    if NOISE_CONFIG["gaussian"]:
        canvas = add_gaussian_noise(canvas, rng, sigma=prng.uniform(4, 14))
    if NOISE_CONFIG["occlusion"]:
        canvas = add_occlusion(canvas, rng, n=prng.randint(2, 5))
    if NOISE_CONFIG["jpeg_artifact"]:
        canvas = add_jpeg_artifacts(canvas, quality=prng.randint(18, 40))

    return canvas, ground_truth


# ─── Ausgabe ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_path = os.path.join(OUTPUT_DIR, IMAGE_NAME)

    scene, gt = generate_scene(N_MARKERS)

    cv2.imwrite(img_path, scene, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    print(f"\nBild gespeichert: {img_path}")

    cv2.namedWindow("ArUco Szene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ArUco Szene", 1400, 900)
    cv2.imshow("ArUco Szene", scene)
    print("Beliebige Taste drücken zum Schließen.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()