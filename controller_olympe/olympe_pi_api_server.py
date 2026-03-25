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
drone_connect_tries = 5  # Number of connection attempts
drone = olympe.Drone(DRONE_IP)
telemetry = {}


def telemetry_loop():
    global telemetry
    while running:
        state = drone.get_state(FlyingStateChanged())
        if state:
            telemetry.update(state)
        telemetry.update(state)
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
        print("Disconnecting drone...")
        drone.disconnect()
        print("Drone disconnected.")
    except Exception as e:
        print(f"Error during shutdown: {e}")
        pass


@atexit.register
def on_exit():
    shutdown()


def main():
    tries = 0
    while tries < drone_connect_tries:
        try:
            drone.connect()
            break
        except Exception as e:
            print(f"Connection attempt {tries+1} failed: {e}")
            tries += 1
            time.sleep(2)
    threading.Thread(target=telemetry_loop, daemon=True).start()

    print(f"Server running at http://{HTTP_HOST}:{HTTP_PORT}")

    # Define all routes
    @app.route("/api/key_down", methods=["POST"])
    def api_key_down():
        # Dummy implementation
        return jsonify(ok=True)

    @app.route("/api/key_up", methods=["POST"])
    def api_key_up():
        # Dummy implementation
        return jsonify(ok=True)

    @app.route("/api/safety/takeoff", methods=["GET"])
    def api_safety_takeoff_get():
        # Dummy implementation
        return jsonify(enabled=True, hold_s=3.0)

    @app.route("/api/logging/telemetry", methods=["GET"])
    def api_telemetry_log_status():
        # Dummy implementation
        return jsonify(enabled=True, path="/path/to/telemetry/log")

    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)
    print(f"Server running at http://{HTTP_HOST}:{HTTP_PORT}")


if __name__ == "__main__":
    main()
