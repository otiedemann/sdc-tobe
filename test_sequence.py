from djitellopy import Tello
import time

tello = Tello()

try:
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    if tello.get_battery() < 25:
        raise Exception("Battery too low for flight test")

    print("Takeoff")
    tello.takeoff()
    time.sleep(2)

    print("Forward 30 cm")
    tello.move_forward(30)
    time.sleep(1)

    print("Rotate 180°")
    tello.rotate_clockwise(180)
    time.sleep(1)

    print("Backward 30 cm")
    tello.move_back(30)
    time.sleep(1)

    print("Land")
    tello.land()
except Exception as e:
    print("Error:", e)
    print("Trying to land safely...")
    try:
        tello.land()
    except:
        pass
finally:
    tello.end()
