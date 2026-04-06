Überblick der Architektur (Weiterentwicklung von pi_position)

Die bestehende pi_position war bereits eine sehr solide Grundlage:

stabile ArUco-Erkennung
funktionierende Positionsbestimmung
durchdachter Runtime-Code (UDP, Debug, Streams)
direkt einsatzfähig auf echter Hardware

Darauf aufbauend haben wir die Architektur erweitert, um zusätzliche Fähigkeiten wie Bewegungsintegration, zeitlich korrekte Fusion und Prädiktion zu ermöglichen.

Neue Struktur (aufbauend auf bestehender Lösung)

Die ursprüngliche Funktionalität bleibt vollständig erhalten und wurde ergänzt durch zusätzliche Module:

pi_position_core.py
bleibt das Herzstück für visuelle Positionsbestimmung (weitgehend unverändert, nur strukturell ausgelagert)
motion_estimator.py
ergänzt die visuelle Lösung um eine kontinuierliche Bewegungsschätzung (IMU / interne Sensoren)
fusion.py
kombiniert Motion und Vision zeitlich korrekt
prediction.py
extrapoliert die Position auf den aktuellen Zeitpunkt
main.py
enthält weiterhin den bestehenden Runtime-Code und orchestriert jetzt zusätzlich die neue Pipeline
Wie das System jetzt arbeitet
1. Bewährte visuelle Position (pi_position_core)

Die bestehende ArUco-basierte Positionsbestimmung liefert weiterhin:

absolute Position im Raum
robuste Referenzpunkte
optional Richtung/Yaw

Diese bleibt die primäre absolute Referenz.

2. Ergänzung durch Motion

Neu ist die kontinuierliche Bewegungsschätzung:

läuft mit hoher Rate (z. B. 25 Hz)
basiert auf Drohnenbewegung (Velocity + Yaw)
hält einen Verlauf der letzten ~2 Sekunden

Dadurch entsteht eine glatte und hochfrequente Positionsschätzung, auch wenn gerade keine Marker sichtbar sind.

3. Zeitlich korrekte Fusion

Die visuelle Messung kommt oft verzögert.
Statt sie einfach „jetzt“ anzuwenden, wird sie:

dem passenden Zeitpunkt in der Vergangenheit zugeordnet
dort mit Motion kombiniert (Kalman-Update)
danach wird die Bewegung bis heute neu berechnet

Das ist ein zentraler Fortschritt gegenüber einer direkten Überschreibung.

4. Prädiktion auf „jetzt“

Da sowohl Vision als auch Fusion leicht verzögert sind, wird der Zustand:

auf den aktuellen Zeitpunkt extrapoliert
unter Annahme konstanter Geschwindigkeit oder per Fit

Zusätzlich wird eine grobe Unsicherheit geschätzt.

Vorteile der Erweiterung
Bestehende Stärken bleiben erhalten
die ArUco-basierte Positionsbestimmung bleibt unverändert nutzbar
der Runtime-Code (UDP, Debug, Streams) bleibt bestehen
das System bleibt sofort einsatzfähig
Höhere Robustheit
funktioniert auch bei kurzzeitig fehlenden Markern
glattere Bewegungsschätzung
weniger Sprünge im Tracking
Zeitlich konsistente Fusion
Vision wird am richtigen Zeitpunkt eingebracht
keine „Sprünge“ durch verspätete Messungen
bessere Gesamtgenauigkeit
Bessere Erweiterbarkeit

Neue Features können jetzt sauber ergänzt werden:

weitere Sensoren (z. B. IMU direkt)
bessere Filter (EKF/UKF)
alternative Vision-Ansätze
Mapping / SLAM später möglich
Wie die Module verwendet werden
Motion
motion_estimator.update_body_frame(...)
Vision
vision_processor.process_frame(...)
Fusion
fuse_delayed_vision_update(...)
Prediction
predict_to_now(...)

Die main.py verbindet diese Bausteine und übernimmt weiterhin:

Kamera-Handling
UDP-Kommunikation
Debug / Preview
Was noch offen ist

Die Architektur ist jetzt sauber vorbereitet, aber einige Dinge können noch weiter verbessert werden:

echtes input-Modul für Telemetrie
robustere Outlier-Erkennung in der Fusion
genauere Unsicherheitsmodellierung
optimierte Prädiktion (z. B. dynamische Modelle)
klarere Datentypen zwischen Modulen
