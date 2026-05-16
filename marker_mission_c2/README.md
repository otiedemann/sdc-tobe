# marker_mission_c2 — local command & control for multiple FCs

Aggregates telemetry + video from up to N marker_mission flight
controllers (default: `flightctrl1..6`, port 8080 each), broadcasts
mission control across the fleet, mirrors arena / tune / mission-script
editing with per-FC checkbox push, and keeps a central library of
calibration `.npz` files synced from every FC.

Local-only for now — binds to `127.0.0.1:8090`, no auth.

## Run

```bash
pip install -r marker_mission_c2/requirements.txt   # first time only

python -m marker_mission_c2                       # uses config.example.json
python -m marker_mission_c2 --config my-cfg.json  # explicit config
MARKER_MISSION_C2_CONFIG=path python -m marker_mission_c2
```

Then open <http://127.0.0.1:8090/>.

## Pages

- `/` — overview grid: one card per FC with phase / battery / position /
  per-FC video toggle / per-FC start+stop / inline script editor.
- `/arena` — JSON arena editor; load from any FC, push to any subset.
- `/tune` — current-value tune form; load from any FC, apply to any
  subset (optional `also_save` checkbox to persist on each).
- `/calibrate` — iframe grid of every FC's native `/calibrate` page,
  plus a central library that syncs `.npz` files from every FC and lets
  you push any of them to any FC (use when a drone moves between FCs).
- `/scripts` — fleet mission-script editor: push as draft, push-and-start,
  or stop across selected FCs.

## Emergency land

- Big red `EMERGENCY LAND ALL` button at the top of every page.
- Page-wide `keydown` handler — default key `?`, configurable via
  `emergency_land_key` in the config. Suppressed when focus is inside
  an `<input>`, `<textarea>`, or `contenteditable` element so typing a
  `?` into the script editor doesn't land the swarm.
- Both fan out `POST /api/stop` to every configured FC in parallel; the
  resulting banner reports per-FC ack tally.

## Config

See [`config.example.json`](config.example.json). Schema:

| key                          | meaning                                              |
| ---------------------------- | ---------------------------------------------------- |
| `bind.host` / `bind.port`    | Where the C2 listens (default `127.0.0.1:8090`).    |
| `fcs[]`                      | FCs to manage (`name`, `host`, `port`).             |
| `emergency_land_key`         | One-character key for page-wide emergency land.     |
| `state_poll_hz`              | C2 → FC `/api/state` poll rate (1..10).             |
| `ui_refresh_hz`              | Browser → C2 `/api/c2/overview` poll rate.          |
| `calibration_sync_seconds`   | How often C2 pulls calibration files from each FC.  |
| `fc_request_timeout_s`       | Per-call HTTP timeout for FC requests.              |
| `library_dir`                | Where mirrored calibration `.npz` files live.       |

Resolution order: `--config` flag → `$MARKER_MISSION_C2_CONFIG` →
`./marker_mission_c2/config.json` → `./marker_mission_c2/config.example.json`.

## marker_mission API extensions used by the C2

The C2 mostly speaks the existing marker_mission HTTP surface. The
following endpoints were **added** to marker_mission/ui.py to support
the calibration library:

| Route                                          | Method | Use                                              |
| ---------------------------------------------- | ------ | ------------------------------------------------ |
| `/api/calibrate/files`                         | GET    | List `.npz` files (with scalar metadata).        |
| `/api/calibrate/files/<name>`                  | GET    | Download a raw `.npz`.                           |
| `/api/calibrate/files/<name>`                  | POST   | Upload a `.npz` (validated as NumPy archive).    |
| `/api/calibrate/files/<name>`                  | DELETE | Remove a `.npz`.                                 |
| `/api/identity`                                | GET    | `{hostname, drone_serial, drone_connected, ...}` |

Filenames must match `^anafi_[A-Za-z0-9_\-]+_[A-Za-z0-9]+\.npz$` —
the same pattern `CalibrationStore` writes. Uploads larger than 1 MiB
are rejected.

## Dev without 6 real FCs

Use [`tools/fake_fc.py`](tools/fake_fc.py):

```bash
for p in 9001 9002 9003 9004 9005 9006; do
  python -m marker_mission_c2.tools.fake_fc \
    --port $p --serial FAKE$p &
done
```

Then create `marker_mission_c2/config.json`:

```json
{
  "bind": {"host": "127.0.0.1", "port": 8090},
  "fcs": [
    {"name": "flightctrl1", "host": "127.0.0.1", "port": 9001},
    {"name": "flightctrl2", "host": "127.0.0.1", "port": 9002},
    {"name": "flightctrl3", "host": "127.0.0.1", "port": 9003},
    {"name": "flightctrl4", "host": "127.0.0.1", "port": 9004},
    {"name": "flightctrl5", "host": "127.0.0.1", "port": 9005},
    {"name": "flightctrl6", "host": "127.0.0.1", "port": 9006}
  ],
  "emergency_land_key": "?"
}
```

Run `python -m marker_mission_c2` and exercise the smoke-test in
`docs/implementation_plan.md`.

## Smoke-test checklist

1. Open `/`. All cards populate within ~1 s.
2. Kill one fake FC. Its card flips to a red dot within ~3 s.
3. Click `Show video` on one card; the `<img>` appears. `Hide video`
   removes it (closes the MJPEG socket).
4. `/arena` → `Load from flightctrl3` → uncheck flightctrl1 →
   `Push to selected` → 5/6 ok.
5. `/calibrate` → push a row from the library → verify it appears in
   the target FC's `data_dir/calibrations/`.
6. `/` → press `?` with focus on the page body. Red banner +
   confirmation banner with ack tally.
7. **Critical regression check:** focus a script `<textarea>`, type `?`.
   The `?` lands in the textarea; no banner.
8. Top button **EMERGENCY LAND ALL** does the same as the key.

## Architecture (one paragraph)

The Flask app runs in the main thread (Werkzeug threaded=True). A
single daemon thread, `c2-loop`, owns an asyncio event loop with one
`httpx.AsyncClient` and per-FC poll tasks. Flask handlers bridge into
the loop via `asyncio.run_coroutine_threadsafe(...).result(timeout)`.
Per-FC state is cached in an `FCState` dataclass, mutated under per-FC
locks, and read by Flask via a deep-copied `snapshot()` so the UI never
sees a mid-update dict.
