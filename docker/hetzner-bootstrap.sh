#!/usr/bin/env bash
# hetzner-bootstrap.sh — go from a fresh Ubuntu 22.04 GPU server
# to a running sdc-sphinx container in one command.
#
# Usage:
#
#   curl -sSL https://raw.githubusercontent.com/otiedemann/sdc-tobe/main/docker/hetzner-bootstrap.sh \
#     | sudo bash -s -- ghcr.io/otiedemann/sdc-sphinx:latest
#
#   ↑ pass the image as $1.
#
# Env vars (optional — script prompts via /dev/tty if missing):
#   GHCR_USER       Username for ghcr.io login (default: otiedemann)
#   GHCR_PAT        GitHub PAT with read:packages scope (required for private)
#   TS_AUTHKEY      Tailscale auth key (recommended)
#   TS_HOSTNAME     Tailscale device name (default: $(hostname))
#   SDC_ANAFI_IP    Override sim drone IP (default: 10.202.0.1)
#
# Tested on:
#   Hetzner Cloud GPU (GEX44, RTX 4000 Ada)
#   Hetzner Dedicated AX-line with NVIDIA cards
#   Lambda Labs A10/A100 instances
#   DataCrunch A10
#
# Idempotent — safe to re-run; will detect existing installations.

set -euo pipefail

# ── Args ───────────────────────────────────────────────────────────
IMAGE="${1:-ghcr.io/otiedemann/sdc-sphinx:latest}"
SERVICE_NAME="sdc-sphinx"
CONTAINER_NAME="sdc-sphinx"

# ── Helpers ────────────────────────────────────────────────────────
log()  { printf '\n[bootstrap] %s\n' "$*"; }
warn() { printf '\n[bootstrap] WARNING: %s\n' "$*" >&2; }
err()  { printf '\n[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

require_root() {
    [ "$(id -u)" -eq 0 ] || err "Run as root (use sudo)."
}

ask() {
    # ask <var-name> <prompt> [silent]
    local var="$1" prompt="$2" silent="${3:-}"
    if [ -n "${!var:-}" ]; then return 0; fi
    if [ ! -t 0 ] && [ ! -e /dev/tty ]; then
        warn "$var not set and no TTY — set it via env var and re-run."
        return 1
    fi
    if [ -n "$silent" ]; then
        printf '%s: ' "$prompt" > /dev/tty
        IFS= read -r -s "$var" < /dev/tty
        printf '\n' > /dev/tty
    else
        printf '%s: ' "$prompt" > /dev/tty
        IFS= read -r "$var" < /dev/tty
    fi
    export "$var"
}

# ── Preflight ──────────────────────────────────────────────────────
require_root
log "image: $IMAGE"
log "OS check: $(. /etc/os-release && echo "$PRETTY_NAME")"

if [ ! -e /etc/os-release ] || ! grep -q '^ID=ubuntu' /etc/os-release; then
    warn "Not Ubuntu — script is tested on 22.04. Continuing anyway."
fi

# ── NVIDIA driver ──────────────────────────────────────────────────
log "checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "installing NVIDIA driver (this is the slow part — ~3 min)"
    apt-get update
    apt-get install -y --no-install-recommends \
        ubuntu-drivers-common pciutils
    ubuntu-drivers autoinstall
    log "driver installed — a reboot may be required for the kernel module"
    log "if nvidia-smi fails after this script, run: reboot"
else
    log "NVIDIA driver already installed:"
    nvidia-smi -L
fi

# ── Docker ─────────────────────────────────────────────────────────
log "checking Docker"
if ! command -v docker >/dev/null 2>&1; then
    log "installing Docker"
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
else
    log "Docker already installed: $(docker --version)"
fi

# ── NVIDIA Container Toolkit ───────────────────────────────────────
log "checking NVIDIA Container Toolkit"
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    log "installing NVIDIA Container Toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y --no-install-recommends nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
else
    log "NVIDIA Container Toolkit already installed"
fi

# Sanity check the GPU passthrough — if this fails, drone spawn won't work
log "verifying GPU passthrough"
if ! docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 \
        nvidia-smi -L > /tmp/gpu-check 2>&1; then
    warn "docker --gpus all couldn't reach the GPU:"
    cat /tmp/gpu-check >&2
    warn "If you just installed the driver, REBOOT and re-run this script."
    err  "GPU not reachable — aborting before container start."
fi
log "GPU passthrough OK"

# ── Credentials & config ───────────────────────────────────────────
GHCR_USER="${GHCR_USER:-otiedemann}"
ask GHCR_PAT      "GitHub PAT (read:packages) for ghcr.io" silent
ask TS_AUTHKEY    "Tailscale auth key (tskey-auth-...; ENTER to skip)"
TS_HOSTNAME="${TS_HOSTNAME:-$(hostname)}"
SDC_ANAFI_IP="${SDC_ANAFI_IP:-10.202.0.1}"

# ── GHCR login ─────────────────────────────────────────────────────
if [ -n "${GHCR_PAT:-}" ]; then
    log "logging in to ghcr.io as $GHCR_USER"
    echo "$GHCR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
fi

# ── Pull image ─────────────────────────────────────────────────────
log "pulling $IMAGE (this can take ~5 min for first pull, image is ~6 GB)"
docker pull "$IMAGE"

# ── Stop any existing instance ─────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "removing existing $CONTAINER_NAME container"
    docker rm -f "$CONTAINER_NAME"
fi

# ── systemd unit ───────────────────────────────────────────────────
log "writing systemd unit"
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=SDC Sphinx simulator (docker)
After=docker.service network-online.target
Requires=docker.service

[Service]
Restart=always
RestartSec=10
TimeoutStartSec=300

ExecStartPre=-/usr/bin/docker rm -f ${CONTAINER_NAME}
ExecStart=/usr/bin/docker run --rm --name ${CONTAINER_NAME} \\
    --gpus all \\
    --privileged \\
    --cap-add NET_ADMIN --cap-add SYS_ADMIN \\
    -p 8090:8090 \\
    -p 9080:9080 \\
    -p 9081:9081 \\
    -e TS_AUTHKEY=${TS_AUTHKEY:-} \\
    -e TS_HOSTNAME=${TS_HOSTNAME} \\
    -e SDC_ANAFI_IP=${SDC_ANAFI_IP} \\
    -e SDC_GIT_PULL=1 \\
    -e SDC_BRANCH=main \\
    -v sphinx-state:/home/sdc/.parrot-sphinx \\
    -v sphinx-logs:/home/sdc/sdc-tobe/sphinx-control/logs \\
    ${IMAGE}

ExecStop=/usr/bin/docker stop ${CONTAINER_NAME}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"

log "starting ${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}.service"

sleep 3
systemctl --no-pager -l status "${SERVICE_NAME}.service" | head -25 || true

# ── Wait for sphinx-control to come up ─────────────────────────────
log "waiting up to 90 s for sphinx-control on port 8090"
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8090/api/system >/dev/null 2>&1; then
        log "sphinx-control is up after ${i}×3 s"
        break
    fi
    sleep 3
    if [ "$i" = "30" ]; then
        warn "sphinx-control didn't respond on :8090 within 90 s"
        warn "check: journalctl -u ${SERVICE_NAME} -f"
        warn "or:    docker logs ${CONTAINER_NAME}"
    fi
done

# ── Done ───────────────────────────────────────────────────────────
PUBLIC_IP="$(curl -s ifconfig.me 2>/dev/null || echo '<this-host>')"
cat <<EOF

===============================================================
  SDC Sphinx is running.

  Reach the dashboard at:
    http://${PUBLIC_IP}:8090/        (public, if firewall allows)
EOF
if [ -n "${TS_AUTHKEY:-}" ]; then
    cat <<EOF
    http://${TS_HOSTNAME}.<your-tailnet>.ts.net:8090/   (via Tailscale)
EOF
fi
cat <<EOF

  Manage the service:
    systemctl status   ${SERVICE_NAME}
    systemctl restart  ${SERVICE_NAME}
    journalctl -u      ${SERVICE_NAME} -f
    docker logs        ${CONTAINER_NAME} -f

  Update to a newer image:
    docker pull ${IMAGE}
    systemctl restart ${SERVICE_NAME}

  Stop & remove:
    systemctl disable --now ${SERVICE_NAME}
    docker rm -f ${CONTAINER_NAME}
    rm /etc/systemd/system/${SERVICE_NAME}.service
===============================================================
EOF
