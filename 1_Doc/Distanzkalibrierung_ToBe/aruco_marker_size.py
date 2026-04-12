import cv2
import numpy as np
import sys
import os
import re
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def calculate_marker_size(corners):
    """
    Berechnet die Größe eines Aruco-Markers in Pixel basierend auf den Ecken.
    Verwendet die durchschnittliche Seitenlänge des Markers.
    """
    # Ecken sind im Format (4, 2) für die 4 Ecken
    corners = np.array(corners).reshape(4, 2)
    
    # Berechne die Seitenlängen
    side1 = np.linalg.norm(corners[0] - corners[1])
    side2 = np.linalg.norm(corners[1] - corners[2])
    side3 = np.linalg.norm(corners[2] - corners[3])
    side4 = np.linalg.norm(corners[3] - corners[0])
    
    # Durchschnittliche Seitenlänge
    avg_side = (side1 + side2 + side3 + side4) / 4
    return avg_side

def parse_distance_from_filename(filename):
    """
    Extrahiert die Entfernung in Metern aus dem Dateinamen.
    Erwartet ein Format wie 'video_5m.mp4', wo 5 die Entfernung ist.
    """
    # Entferne die Erweiterung
    name = os.path.splitext(filename)[0]
    # Suche nach einem Muster wie '_Xm' oder 'Xm'
    match = re.search(r'(\d+)m', name)
    if match:
        return float(match.group(1))
    else:
        print(f"Fehler: Entfernung konnte nicht aus dem Dateinamen '{filename}' extrahiert werden. Erwartet Format wie 'video_5m.mp4'.")
        return None

def process_video(video_path, known_size_mm):
    """
    Verarbeitet das Video und berechnet die durchschnittliche Marker-Größe in Pixel.
    """
    # Aruco Dictionary laden (hier 4x4_50, kann angepasst werden)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Video laden
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Video {video_path} konnte nicht geöffnet werden.")
        return None
    
    marker_sizes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Marker erkennen
        corners, ids, rejected = detector.detectMarkers(frame)
        
        if ids is not None:
            for corner in corners:
                size = calculate_marker_size(corner)
                marker_sizes.append(size)
    
    cap.release()
    
    if marker_sizes:
        avg_size = np.mean(marker_sizes)
        return avg_size
    else:
        print(f"Keine Aruco-Marker im Video {video_path} gefunden.")
        return None

if __name__ == "__main__":
    # Standard-Pfad zum Ordner (kann hier bearbeitet werden)
    DEFAULT_FOLDER_PATH = "/home/mer/Dokumente/Python/Distanzkalibrierung"  # Ändere diesen Pfad nach Bedarf
    # Bekannte Marker-Größe in mm (kann hier bearbeitet werden)
    DEFAULT_KNOWN_SIZE_MM = 170.0  # 17 cm = 170 mm, ändere nach Bedarf
    
    if len(sys.argv) == 1:
        # Keine Argumente gegeben, verwende Standardwerte
        folder_path = DEFAULT_FOLDER_PATH
        known_size_mm = DEFAULT_KNOWN_SIZE_MM
    elif len(sys.argv) == 2:
        # Nur bekannte_größe_mm gegeben, verwende Standard-Pfad
        folder_path = DEFAULT_FOLDER_PATH
        known_size_mm = float(sys.argv[1])
    elif len(sys.argv) == 3:
        # Ordner-Pfad und bekannte_größe_mm gegeben
        folder_path = sys.argv[1]
        known_size_mm = float(sys.argv[2])
    else:
        print("Verwendung:")
        print("  python aruco_marker_size.py  # Verwendet Standard-Pfad und -Größe")
        print("  python aruco_marker_size.py <bekannte_größe_mm>  # Verwendet Standard-Pfad")
        print("  python aruco_marker_size.py <ordner_pfad> <bekannte_größe_mm>")
        print("Beispiel:")
        print("  python aruco_marker_size.py")
        print("  python aruco_marker_size.py 170")
        print("  python aruco_marker_size.py /pfad/zum/ordner 170")
        sys.exit(1)
    
    # Alle .mp4-Dateien im Ordner finden
    mp4_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    if not mp4_files:
        print(f"Keine MP4-Dateien im Ordner {folder_path} gefunden.")
        sys.exit(1)
    
    # Listen für Kalibrierung sammeln
    collected_distances = []
    collected_pixel_sizes = []
    
    for video_path in mp4_files:
        filename = os.path.basename(video_path)
        distance_m = parse_distance_from_filename(filename)
        if distance_m is None:
            continue  # Überspringe Dateien, bei denen die Entfernung nicht extrahiert werden kann
        
        avg_size = process_video(video_path, known_size_mm)
        
        if avg_size is not None:
            print(f"Datei: {filename}")
            print(f"Entfernung: {distance_m} Meter")
            print(f"Durchschnittliche Marker-Größe: {avg_size:.2f} Pixel")
            print("-" * 40)
            
            # Sammle Daten für Kalibrierung
            collected_distances.append(distance_m)
            collected_pixel_sizes.append(avg_size)
    
    # Kalibrierungsfunktion erstellen
    if collected_distances and collected_pixel_sizes:
        # Potenzfunktion: Fitte y = a * x^b, wo x = 1/pixel, y = distance
        inv_pixels = np.array([1 / p for p in collected_pixel_sizes])
        distances = np.array(collected_distances)
        
        # Logarithmische Transformation für lineare Regression
        log_inv_pixels = np.log(inv_pixels)
        log_distances = np.log(distances)
        
        # Lineare Regression auf log-Daten
        coeffs = np.polyfit(log_inv_pixels, log_distances, 1)
        b, log_a = coeffs  # y = b * x + log_a, also a = exp(log_a)
        a = np.exp(log_a)
        
        print(f"\nPotenzfunktion gefittet: Abstand (m) = {a:.4f} * (1/Pixel)^{b:.4f}")
        
        # Definiere die Funktion
        def estimate_distance_from_pixels(pixel_size):
            if pixel_size == 0:
                return float('inf')
            inv_p = 1 / pixel_size
            return a * (inv_p ** b)
        
        # Beispiel: Geschätzte Entfernung für den Durchschnitt der Pixel-Größen
        avg_pixel = np.mean(collected_pixel_sizes)
        estimated_dist = estimate_distance_from_pixels(avg_pixel)
        print(f"Beispiel: Für durchschnittliche {avg_pixel:.2f} Pixel geschätzte Entfernung: {estimated_dist:.2f} m")
        
        # Graphische Ausgabe der Kalibrierungsfunktion
        plt.figure(figsize=(8, 6))
        plt.scatter(collected_pixel_sizes, collected_distances, label='Datenpunkte', color='blue')
        
        # Regressionslinie: Verwende die Potenzfunktion
        pixel_line = np.linspace(min(collected_pixel_sizes), max(collected_pixel_sizes), 100)
        dist_line = [estimate_distance_from_pixels(p) for p in pixel_line]
        plt.plot(pixel_line, dist_line, color='red', label=f'Potenzfunktion: Abstand = {a:.4f} * (1/Pixel)^{b:.4f}')
        
        plt.xlabel('Anzahl der Pixel')
        plt.ylabel('Entfernung')
        plt.title('Kalibrierung: Entfernung vs. Pixel-Größe')
        plt.legend()
        plt.grid(True)
        
        # Y-Achse beginnt bei 0
        plt.ylim(bottom=0)
        # X-Achse von 0 bis 200 Pixel
        plt.xlim(0, 200)
        
        # Füge die Funktion der Trendlinie als Text im Diagramm hinzu
        func_text = f'Abstand = {a:.4f} * (1/Pixel)^{b:.4f}'
        plt.text(0.05, 0.95, func_text, 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Lineare Achsen (entferne logarithmische Skalierung)
        # Beschriftung der y-Achse in der Form "x.xxm"
        def format_distance(value, tick_number):
            return f"{value:.1f}m"
        
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_distance))
        
        # Speichere das Diagramm als Bild
        plt.savefig('kalibrierung.png')
        print("Diagramm gespeichert als 'kalibrierung.png' im aktuellen Verzeichnis.")
        
        # Zeige das Diagramm sofort an
        plt.show()
        plt.close()  # Schließe die Figur nach dem Anzeigen
        
        # Die Funktion kann nun verwendet werden, z.B.:
        # dist = estimate_distance_from_pixels(50)  # Für 50 Pixel
    else:
        print("Nicht genügend Daten für Kalibrierung gesammelt.")