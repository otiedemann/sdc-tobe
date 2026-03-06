from djitellopy import Tello
import time

tello = Tello()

try:
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    tello.streamoff()
    print("Taking off...")
    tello.takeoff()
    time.sleep(3)
    print("Landing...")
    tello.land()
    print("Done.")
except Exception as e:
    print("Error:", e)
    try:
        tello.land()
    except:
        pass
finally:
    tello.end()
