import cv2

# IP-Adresse deiner Handy-App (z.B. IP Webcam oder DroidCam)
# Hier deine Handy-IP eintragen (z.B. "http://192.168.1.50:8080/video")
url = 1


def main():
    print(f"Suche Stream auf: {url}")
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Fehler: Stream konnte nicht geöffnet werden.")
        print("Tipp: Prüfe IP und WLAN (Handy und Laptop müssen im selben Netz sein).")
        return

    # Fenster festlegen
    cv2.namedWindow('Swarm Cam - Monitoring', cv2.WINDOW_NORMAL)

    print("Stream läuft! Drücke 'q' auf der Tastatur zum Beenden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Verbindung zum Handy verloren.")
            break

        # Bild anzeigen
        cv2.imshow('Swarm Cam - Monitoring', frame)

        # Abbrechen mit 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
