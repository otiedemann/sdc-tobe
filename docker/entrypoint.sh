#!/usr/bin/env bash
# Entrypoint for the sdc-sphinx container.
#
# Responsibilities:
#   1. Start Xvfb (virtual X server) so UE4 has a display surface.
#   2. Start tailscaled in userspace-networking mode (auth via TS_AUTHKEY
#      if provided; otherwise leave it idle for the operator to attach).
#   3. Pull latest sdc-tobe (optional, controlled by SDC_GIT_PULL).
#   4. Hand off to sphinx-control (or whatever CMD was given).
#
# Environment variables:
#   TS_AUTHKEY      Tailscale auth key (tskey-...). If unset, tailscale
#                   stays disabled — connect manually with `docker exec`.
#   TS_HOSTNAME     Hostname Tailscale advertises (default: container's).
#   SDC_GIT_PULL    "1" to git pull the repo on every start (default 0).
#   SDC_BRANCH      Branch to checkout if SDC_GIT_PULL=1 (default main).
#   DISPLAY         Override the Xvfb display (default :99).
#   XVFB_RES        Xvfb resolution (default 1280x1024x24).
#
# On error any of the steps print to stderr; the script never `set -e`s
# the entire flow because we want sphinx-control to come up even if
# tailscale auth fails (operator can still reach it via -p 8090).

set -u

DISPLAY="${DISPLAY:-:99}"
XVFB_RES="${XVFB_RES:-1280x1024x24}"
SDC_GIT_PULL="${SDC_GIT_PULL:-0}"
SDC_BRANCH="${SDC_BRANCH:-main}"
REPO=/home/sdc/sdc-tobe

log() { printf '\n[entrypoint] %s\n' "$*"; }

# ── 1. Xvfb ─────────────────────────────────────────────────────────────
log "starting Xvfb on ${DISPLAY} (res=${XVFB_RES})"
# -screen 0 W×H×D ; -ac disables host-based access control (the
# container is already isolated); +extension GLX/RANDR/RENDER to make
# UE4 happy.
Xvfb "${DISPLAY}" -screen 0 "${XVFB_RES}" -ac \
     +extension GLX +extension RANDR +extension RENDER \
     -nolisten tcp >/var/log/xvfb.log 2>&1 &
XVFB_PID=$!
export DISPLAY

# Wait briefly for Xvfb to come up
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then break; fi
    sleep 0.5
done
if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "[entrypoint] WARNING: Xvfb on ${DISPLAY} not responding (see /var/log/xvfb.log)" >&2
fi

# ── 2. Tailscale ────────────────────────────────────────────────────────
if command -v tailscaled >/dev/null 2>&1; then
    log "starting tailscaled (userspace-networking)"
    mkdir -p /var/lib/tailscale
    tailscaled --tun=userspace-networking --statedir=/var/lib/tailscale \
               --socket=/var/run/tailscale/tailscaled.sock \
               >/var/log/tailscaled.log 2>&1 &
    TAILSCALED_PID=$!
    sleep 1
    if [ -n "${TS_AUTHKEY:-}" ]; then
        log "tailscale up (hostname=${TS_HOSTNAME:-$(hostname)})"
        tailscale --socket=/var/run/tailscale/tailscaled.sock up \
            --authkey="${TS_AUTHKEY}" \
            --hostname="${TS_HOSTNAME:-$(hostname)}" \
            --accept-routes \
            || echo "[entrypoint] WARNING: tailscale up failed (continuing)" >&2
    else
        log "TS_AUTHKEY not set — tailscale running but not authed."
        log "Attach later with:  docker exec -it <ctr> tailscale up --authkey=tskey-..."
    fi
else
    log "tailscale binary missing; skipping"
fi

# ── 3. Repo refresh (optional) ─────────────────────────────────────────
if [ "${SDC_GIT_PULL}" = "1" ]; then
    log "pulling sdc-tobe (branch=${SDC_BRANCH})"
    su sdc -c "cd ${REPO} && git fetch && git checkout ${SDC_BRANCH} && git pull --ff-only" \
        || echo "[entrypoint] WARNING: git pull failed (continuing)" >&2
fi

# sphinx-control reads config.yaml; the repo only ships
# config.example.yaml. On first container start, copy the example
# in place so the launcher has something to load — without this the
# FC falls back to controller_unified's hardwired 192.168.42.1
# (the real-Anafi-over-Wi-Fi IP, useless inside the container).
if [ ! -f "${REPO}/sphinx-control/config.yaml" ] \
   && [ -f "${REPO}/sphinx-control/config.example.yaml" ]; then
    log "creating sphinx-control/config.yaml from example"
    cp "${REPO}/sphinx-control/config.example.yaml" \
       "${REPO}/sphinx-control/config.yaml"
    chown sdc:sdc "${REPO}/sphinx-control/config.yaml"
fi

# Sim-friendly default: point the FC at the simulated Anafi IP
# instead of the real-Anafi-over-Wi-Fi default 192.168.42.1. Sphinx
# 2 with netns mode → 10.202.0.1; older sphinx-control ports mode
# also resolves there. Operator can still override per-start via
# the UI's Anafi-IP input. Set SDC_ANAFI_IP at Pod-launch time to
# pick a different default (e.g. 172.21.0.2 for some setups).
SDC_ANAFI_IP="${SDC_ANAFI_IP:-10.202.0.1}"
if grep -q '^[[:space:]]*anafi_ip:[[:space:]]*null' \
       "${REPO}/sphinx-control/config.yaml" 2>/dev/null; then
    log "patching config.yaml: anafi_ip null → ${SDC_ANAFI_IP}"
    sed -i "s|^\([[:space:]]*\)anafi_ip:[[:space:]]*null|\1anafi_ip: \"${SDC_ANAFI_IP}\"|" \
        "${REPO}/sphinx-control/config.yaml"
fi

# Fresh Docker named volumes are created owned by root, but sphinx-
# control runs as user `sdc` (Sphinx refuses root). When the
# volumes mount over the image's existing dirs the ownership
# resets to root and sdc can't write into them — the launcher then
# fails on env-start with "Permission denied: .../logs/env-XXXXXX".
# Chown the two mount points to sdc on every start; idempotent.
for d in "${REPO}/sphinx-control/logs" /home/sdc/.parrot-sphinx; do
    if [ -d "$d" ]; then
        owner=$(stat -c '%U' "$d" 2>/dev/null || echo unknown)
        if [ "$owner" != "sdc" ]; then
            log "chown $d → sdc:sdc (was $owner)"
            chown -R sdc:sdc "$d" 2>/dev/null || true
        fi
    fi
done

# Make sure both venvs are there (rebuilds happen if the operator
# mounted a fresh repo over /home/sdc/sdc-tobe).
if [ ! -x "${REPO}/sphinx-control/.venv/bin/uvicorn" ]; then
    log "sphinx-control venv missing — rebuilding"
    su sdc -c "python3 -m venv ${REPO}/sphinx-control/.venv \
        && ${REPO}/sphinx-control/.venv/bin/pip install --upgrade pip \
        && ${REPO}/sphinx-control/.venv/bin/pip install \
            fastapi uvicorn pyyaml requests sqlalchemy psutil \
            $(test -f ${REPO}/sphinx-control/requirements.txt && \
              echo -r ${REPO}/sphinx-control/requirements.txt)"
fi

# Repo-level venv for the unified flight-controller (path is hard-
# coded in sphinx-control/config.yaml as flight_controller.python =
# /home/sdc/sdc-tobe/.venv/bin/python). Without this the UI's
# "FC start" button fails with FileNotFoundError on /python.
if [ ! -x "${REPO}/.venv/bin/python" ]; then
    log "flight-controller venv missing — rebuilding (this can take a few minutes; parrot-olympe is large)"
    su sdc -c "python3 -m venv ${REPO}/.venv \
        && ${REPO}/.venv/bin/pip install --upgrade pip wheel \
        && ${REPO}/.venv/bin/pip install -r ${REPO}/controller_unified/requirements.txt"
fi

# ── 3-pre. SSH server ──────────────────────────────────────────────────
# Run sshd inside the container so operators can `ssh root@host -p <vast-port>`
# for live debugging. Vast/RunPod expose a port → 22 mapping by default;
# bare-metal hosts can hit it on the container's local 22 once they're
# already on the host. Generates host keys on first start, then reuses.
#
# Authorised keys come from $SSH_PUBLIC_KEY env var (Vast injects this
# automatically with all the keys you registered on your account) OR
# from /root/.ssh/authorized_keys if you mount one in.
if command -v sshd >/dev/null 2>&1; then
    mkdir -p /var/run/sshd /root/.ssh
    chmod 700 /root/.ssh
    if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
        echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        log "installed SSH key from SSH_PUBLIC_KEY env var"
    fi
    # Generate host keys if they don't exist (first run)
    [ -f /etc/ssh/ssh_host_ed25519_key ] || ssh-keygen -A >/dev/null 2>&1
    # Permit root login via key (no password — disable PasswordAuth)
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
    sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    if ! pgrep -x sshd >/dev/null 2>&1; then
        log "starting sshd on port 22"
        /usr/sbin/sshd -D >/var/log/sshd.log 2>&1 &
    fi
fi

# ── 3a. firmwared ──────────────────────────────────────────────────────
# The parrot-sphinx package ships /usr/bin/firmwared, normally started
# via systemd on bare-metal Ubuntu. In a docker container there's no
# systemd, so we start it ourselves. Without firmwared, every drone
# spawn dies with sphinx complaining it can't reach the firmware
# manager. Run as root (firmwared chroots itself) and keep the
# command line minimal — defaults are fine for headless sim.
if command -v firmwared >/dev/null 2>&1; then
    if ! pgrep -x firmwared >/dev/null 2>&1; then
        log "starting firmwared (binary: $(command -v firmwared))"
        firmwared >/var/log/firmwared.log 2>&1 &
        # Wait briefly for the socket
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            [ -S /var/run/firmwared.sock ] && break
            sleep 0.5
        done
        if [ ! -S /var/run/firmwared.sock ]; then
            echo "[entrypoint] WARNING: /var/run/firmwared.sock didn't appear within 5s — dumping firmwared log:" >&2
            tail -30 /var/log/firmwared.log >&2 || true
            echo "[entrypoint] drone spawn will fail until firmwared is reachable" >&2
        else
            log "firmwared up — socket: /var/run/firmwared.sock"
        fi
    fi
else
    log "firmwared not on PATH — drone spawn will fail (parrot-sphinx incomplete?)"
fi

# ── 3b. Lazy arena build (if image was built with PREBUILD_ARENA=0) ───
# When the image is cross-built on an M1 Mac, the Blender step is
# usually skipped (PREBUILD_ARENA=0) because QEMU emulation makes it
# painfully slow. Detect a missing arena.yml and build it now on
# real x86 hardware — first container start takes ~30 s longer but
# every subsequent start is instant.
if [ ! -f "${REPO}/tools/sphinx-arena/out/arena.yml" ] \
   || [ ! -f "${REPO}/tools/sphinx-arena/out/fbx/aruco_001.fbx" ]; then
    log "arena assets missing — building now (one-time, ~60 s)"
    su sdc -c "cd ${REPO}/tools/sphinx-arena \
        && make pngs && make fbxs && make arena-static \
        && (make logos && make paintings || true) \
        && make yaml" \
        || echo "[entrypoint] WARNING: arena build failed (continuing)" >&2
fi

# ── 4. Hand off to the requested command ───────────────────────────────
case "${1:-sphinx-control}" in
    sphinx-control)
        log "starting sphinx-control on 0.0.0.0:8090 as user sdc"
        # Source parrot-sphinx-setenv.sh so `pysphinx` is importable
        # from sphinx-control (the bundled Python lib lives under
        # /opt/parrot-sphinx/usr/lib/...). Run as 'sdc' (Sphinx
        # refuses root). DISPLAY inherits via env.
        exec su sdc -c ". /opt/parrot-sphinx/usr/bin/parrot-sphinx-setenv.sh \
            && cd ${REPO}/sphinx-control \
            && DISPLAY=${DISPLAY} \
               .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8090"
        ;;
    bash|sh|shell)
        log "dropping to interactive shell as sdc"
        exec su - sdc
        ;;
    *)
        log "running custom command: $*"
        exec "$@"
        ;;
esac
