from models import Vec2

SCENARIOS = {
    "baseline": {
        "duration_s": 240,
        "dt": 0.1,
        "world": {
            "width_m": 12.0,
            "height_m": 8.0,
            "home": Vec2(1.0, 1.0),
            "target": Vec2(10.0, 6.0),
        },
        "drones": [
            {"id": "D1", "start": Vec2(1.0, 1.0)},
            {"id": "D2", "start": Vec2(1.5, 1.2)},
        ],
        "comms": {"latency_ms_mean": 50, "latency_ms_jitter": 20, "packet_loss": 0.02, "dropout_prob_per_tick": 0.0},
    },
    "degraded_link": {
        "duration_s": 240,
        "dt": 0.1,
        "world": {
            "width_m": 12.0,
            "height_m": 8.0,
            "home": Vec2(1.0, 1.0),
            "target": Vec2(10.0, 6.0),
        },
        "drones": [
            {"id": "D1", "start": Vec2(1.0, 1.0)},
            {"id": "D2", "start": Vec2(1.5, 1.2)},
            {"id": "D3", "start": Vec2(1.2, 1.6)},
        ],
        "comms": {"latency_ms_mean": 120, "latency_ms_jitter": 60, "packet_loss": 0.15, "dropout_prob_per_tick": 0.0},
    },
    "dropout": {
        "duration_s": 240,
        "dt": 0.1,
        "world": {
            "width_m": 12.0,
            "height_m": 8.0,
            "home": Vec2(1.0, 1.0),
            "target": Vec2(10.0, 6.0),
        },
        "drones": [
            {"id": "D1", "start": Vec2(1.0, 1.0)},
            {"id": "D2", "start": Vec2(1.5, 1.2)},
            {"id": "D3", "start": Vec2(1.2, 1.6)},
        ],
        "comms": {"latency_ms_mean": 70, "latency_ms_jitter": 35, "packet_loss": 0.05, "dropout_prob_per_tick": 0.0015},
    },
}
