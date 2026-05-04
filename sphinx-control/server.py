"""Sphinx Control — FastAPI web service.

Run with::

    cd sphinx-control
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp config.example.yaml config.yaml   # then edit
    uvicorn server:app --host 0.0.0.0 --port 8090

Or via the provided systemd unit.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from launcher import EnvironmentRequest, LaunchRequest, Launcher
from state import StateStore
from worlds import Registry

logging.basicConfig(
    level=os.getenv("SPHINX_CONTROL_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sphinx-control.server")

ROOT = Path(__file__).resolve().parent


def _load_config() -> dict[str, Any]:
    cfg_path = Path(
        os.getenv("SPHINX_CONTROL_CONFIG", str(ROOT / "config.yaml"))
    )
    if not cfg_path.is_file():
        # Fall back to the example so a fresh checkout boots without
        # surprise. The dashboard banner will show "using example
        # config" so the operator notices.
        cfg_path = ROOT / "config.example.yaml"
        log.warning("No config.yaml found, falling back to config.example.yaml")
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


config = _load_config()
state = StateStore(config.get("state_db", str(ROOT / "sphinx_control.db")))
registry = Registry.from_config(config)
launcher = Launcher(
    config=config,
    registry=registry,
    state=state,
    log_dir=ROOT / "logs",
)
launcher.reconcile_on_startup()


app = FastAPI(title="Sphinx Control", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# Use Jinja2 directly rather than starlette.templating.Jinja2Templates —
# the old TemplateResponse(name, ctx) signature breaks under
# Starlette ≥ 1.0 + Python 3.14, and the new signature requires a
# Request object that we don't otherwise need here. Direct rendering
# is one line and avoids the version drift.
_jinja_env = Environment(
    loader=FileSystemLoader(str(ROOT / "templates")),
    autoescape=select_autoescape(["html"]),
)


def _render(name: str, **ctx: Any) -> HTMLResponse:
    return HTMLResponse(_jinja_env.get_template(name).render(**ctx))


# ─── Pydantic schemas ────────────────────────────────────────────


class SpawnRequest(BaseModel):
    drone_profile: str = Field(..., examples=["anafi"])
    # world_name is accepted for back-compat but ignored — the world
    # belongs to the environment, which must be running before a
    # drone can spawn.
    world_name: str | None = None
    instance_id: int | None = Field(None, ge=1, le=10)
    firmware_url: str | None = None


class EnvSpawnRequest(BaseModel):
    world_name: str = Field(..., examples=["empty", "sdc_arena"])


class StatusOK(BaseModel):
    ok: bool = True


# ─── Page handlers ───────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index():
    return _render(
        "index.html",
        title="Sphinx Control",
        max_drones=launcher.max_drones,
        network_mode=launcher.mode,
        network_warning=launcher.network_warning,
        session_warning=launcher.session_warning,
        dry_run=launcher.dry_run,
    )


# ─── API: environment (UE4 only) ─────────────────────────────────


@app.get("/api/environment")
def api_get_environment():
    env = launcher.current_environment()
    return env.to_dict() if env else None


@app.post("/api/environment", status_code=201)
def api_start_environment(req: EnvSpawnRequest):
    try:
        env = launcher.start_environment(EnvironmentRequest(world_name=req.world_name))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception("environment start failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return env.to_dict()


@app.delete("/api/environment")
def api_stop_environment():
    try:
        launcher.stop_environment()
    except Exception as e:
        log.exception("environment stop failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return StatusOK()


@app.post("/api/environment/restart")
def api_restart_environment():
    try:
        env = launcher.restart_environment()
    except KeyError:
        raise HTTPException(status_code=404, detail="no environment running")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception("environment restart failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return env.to_dict()


# ─── API: drones ─────────────────────────────────────────────────


@app.get("/api/drones")
def api_list_drones():
    return [r.to_dict() for r in launcher.list_drones()]


@app.post("/api/drones", status_code=201)
def api_spawn_drone(req: SpawnRequest):
    try:
        rec = launcher.spawn(LaunchRequest(
            drone_profile=req.drone_profile,
            world_name=req.world_name,
            instance_id=req.instance_id,
            firmware_url=req.firmware_url,
        ))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        # Any other failure (sqlite IntegrityError, OSError on Popen,
        # XAuthority not readable, etc.). Without this catch the dialog
        # in the browser just says "500 Internal Server Error" and the
        # operator has to grep journalctl. Surface the type+message so
        # the alert is actually useful.
        log.exception("spawn failed unexpectedly")
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )
    return rec.to_dict()


@app.delete("/api/drones/{drone_id}")
def api_delete_drone(drone_id: str):
    try:
        launcher.delete(drone_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StatusOK()


@app.post("/api/drones/{drone_id}/stop")
def api_stop_drone(drone_id: str):
    try:
        launcher.stop(drone_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="drone not found")
    return StatusOK()


@app.post("/api/drones/{drone_id}/restart")
def api_restart_drone(drone_id: str):
    try:
        rec = launcher.restart(drone_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="drone not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return rec.to_dict()


@app.post("/api/drones/restart-all")
def api_restart_all():
    return [r.to_dict() for r in launcher.restart_all()]


@app.post("/api/drones/stop-all")
def api_stop_all():
    n = launcher.stop_all()
    return {"stopped": n}


@app.get("/api/drones/{drone_id}/connections")
def api_drone_connections(drone_id: str):
    try:
        return launcher.connections_for(drone_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="drone not found")


# ─── API: worlds, profiles, system ──────────────────────────────


@app.get("/api/worlds")
def api_worlds():
    return [
        {
            "name": w.name,
            "binary": w.binary,
            "description": w.description,
            "config_file": w.config_file,
            "available": w.available,
            "unavailable_reason": w.unavailable_reason,
        }
        for w in registry.list_worlds()
    ]


@app.get("/api/profiles")
def api_profiles():
    return [
        {
            "name": p.name,
            "descriptor": p.descriptor,
            "descriptor_path": p.descriptor_path,
            "description": p.description,
            "available": p.available,
            "unavailable_reason": p.unavailable_reason,
        }
        for p in registry.list_profiles()
    ]


@app.get("/api/system")
def api_system():
    return {
        "max_drones": launcher.max_drones,
        "network_mode": launcher.mode,
        "network_warning": launcher.network_warning,
        "dry_run": launcher.dry_run,
        "session_attach": launcher.session_attach_setting,
        "session_warning": launcher.session_warning,
        "active_session": launcher.session.summary() if launcher.session else None,
        "tailscale": _tailscale_status(),
        "host": _host_info(),
        "sphinx": _sphinx_info(),
    }


# ─── helpers ─────────────────────────────────────────────────────


def _tailscale_status() -> dict[str, Any]:
    cfg = config.get("tailscale", {}) or {}
    if not cfg.get("enabled"):
        return {"enabled": False}
    binary = cfg.get("binary", "/usr/bin/tailscale")
    if not Path(binary).is_file() and shutil.which(binary) is None:
        return {"enabled": True, "available": False, "error": "tailscale binary not found"}
    try:
        out = subprocess.check_output(
            [binary, "status", "--json"], text=True, timeout=3,
        )
        data = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        return {"enabled": True, "available": False, "error": str(e)}
    self_node = data.get("Self") or {}
    addrs = self_node.get("TailscaleIPs") or []
    advertised = self_node.get("AdvertisedRoutes") or []
    return {
        "enabled": True,
        "available": True,
        "tailnet_name": data.get("MagicDNSSuffix", "").rstrip("."),
        "self_addrs": addrs,
        "self_hostname": self_node.get("HostName"),
        "advertised_routes": advertised,
        "online_peers": [
            {"hostname": p.get("HostName"), "addrs": p.get("TailscaleIPs", [])}
            for p in (data.get("Peer") or {}).values()
            if p.get("Online")
        ],
    }


def _host_info() -> dict[str, Any]:
    import platform
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    # Best-effort load average — only on Unix.
    try:
        info["loadavg_1_5_15"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    return info


def _sphinx_info() -> dict[str, Any]:
    binary = (config.get("sphinx", {}) or {}).get("binary", "/usr/bin/sphinx")
    if not Path(binary).is_file():
        return {"available": False, "binary": binary}
    # Sphinx has no --version flag (verified on Sphinx 2.15: it exits
    # non-zero with usage). Probe via its package name instead — same
    # info, no Sphinx invocation needed.
    version = "installed"
    try:
        out = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Version}", "parrot-sphinx"],
            text=True, timeout=3,
        )
        if out.strip():
            version = out.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        # Not Debian/Ubuntu, or parrot-sphinx wasn't installed via apt.
        # The binary still exists (we checked above), so just say so.
        pass
    return {"available": True, "binary": binary, "version": version}


# ─── healthz ─────────────────────────────────────────────────────


@app.get("/healthz")
def healthz():
    return {"ok": True}
