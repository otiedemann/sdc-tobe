import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()

try:
    while True:
        frame = frame_read.frame
        if frame is None:
            continue

        cv2.imshow("Tello", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    tello.streamoff()
    tello.end()
