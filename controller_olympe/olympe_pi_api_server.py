import atexit
import olympe
import os
import threading
import time
from flask import Flask, jsonify, request, send_file
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.Animations import Flip

# Olympe API server (single Anafi drone)
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

app = Flask(__name__)
running = True
DRONE_IP = "192.168.42.1"

drone = olympe.Drone(DRONE_IP)
telemetry = {}


def telemetry_loop():
    global telemetry
    while running:
        telemetry = drone.get_state(FlyingStateChanged())
        time.sleep(0.5)


@app.route("/api/takeoff", methods=["POST"])
def api_takeoff():
    if not drone(TakeOff()).wait().success():
        return jsonify(ok=False, error="takeoff_failed"), 500
    return jsonify(ok=True)

@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(silent=True) or {}
    dx = float(data.get("dx", 0))
    dy = float(data.get("dy", 0))
    dz = float(data.get("dz", 0))
    if not drone(moveBy(dx, dy, dz, 0)).wait().success():
        return jsonify(ok=False, error="move_failed"), 500
    return jsonify(ok=True)


@app.route("/api/flip", methods=["POST"])
def api_flip():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "front").lower()
    flip_map = {
        "front": Flip.direction.front,
        "back": Flip.direction.back,
        "left": Flip.direction.left,
        "right": Flip.direction.right
    }
    if direction not in flip_map:
        return jsonify(ok=False, error="invalid_direction"), 400
    if not drone(Flip(flip_map[direction])).wait().success():
        return jsonify(ok=False, error="flip_failed"), 500
    return jsonify(ok=True)


@app.route("/api/land", methods=["POST"])
def api_land():
    if not drone(Landing()).wait().success():
        return jsonify(ok=False, error="land_failed"), 500
    return jsonify(ok=True)


def shutdown():
    global running
    running = False
    try:
        drone.disconnect()
    except Exception:
        pass


@atexit.register
def on_exit():
    shutdown()


def main():
    drone.connect()
    threading.Thread(target=telemetry_loop, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)
    print(f"Server running at http://{HTTP_HOST}:{HTTP_PORT}")


if __name__ == "__main__":
    main()
