from djitellopy import Tello

tello = Tello()
tello.connect()
print("Connected. Battery:", tello.get_battery(), "%")
tello.end()
