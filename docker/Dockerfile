# SDC26 Sphinx simulator container — Ubuntu 22.04 + Parrot Sphinx + sphinx-control
#
# Base image: NVIDIA CUDA on Ubuntu 22.04. This is what RunPod / Lambda Labs
# / similar GPU cloud providers expect. The host MUST have an NVIDIA GPU
# and the NVIDIA Container Toolkit installed; with `--gpus all`, the GPU
# driver + Vulkan ICD are bind-mounted into the container at runtime.
#
# Sphinx + UE4 cannot truly run "headless" — UE4 needs a Vulkan-capable
# surface — but we don't need a real monitor: Xvfb (a virtual X server)
# gives UE4 a framebuffer to render into. The drone's camera stream is
# served by Sphinx over its own ports, completely separate from the
# UE4 viewport, so an operator never needs to see the UE4 window.
#
# Build:
#   docker build -t sdc-sphinx:latest -f docker/Dockerfile .
#
# Run (RunPod / GPU host):
#   docker run --gpus all --rm -it \
#       -p 8090:8090 -p 9080:9080 \
#       -e TS_AUTHKEY=tskey-... \
#       sdc-sphinx:latest
#
# See docker/README.md for full RunPod / Lambda / local instructions.

ARG BASE_IMAGE=nvidia/cuda:12.2.2-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
# Vulkan loader needs to find the NVIDIA ICD which the container
# toolkit bind-mounts at /usr/share/glvnd/egl_vendor.d/ +
# /usr/share/vulkan/icd.d/.

# ── System packages ─────────────────────────────────────────────────────
# - xvfb : virtual X server so UE4 has a display surface (no real monitor)
# - x11-utils + xdotool : sphinx-control sends F10 to UE4 to hide HMI
# - libvulkan1, mesa-vulkan-drivers, vulkan-tools : Vulkan loader stack
# - libgl1, libglu1-mesa, libxcb1, etc. : UE4's native runtime deps
# - net-tools, iputils-ping : sanity check inside container
# - sudo : the parrot install docs use it; some post-install scripts
#   call it directly, so keep it even though we'll be root inside the
#   container by default
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg lsb-release sudo \
        git openssh-client sshpass \
        xvfb x11-utils xdotool xauth \
        libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libgl1 \
        libglu1-mesa libxcb1 libxcb-xinerama0 libxkbcommon0 libnss3 \
        libasound2 libgbm1 libdbus-1-3 \
        net-tools iputils-ping iproute2 \
        python3 python3-venv python3-pip python3-dev \
        python3-numpy \
        build-essential pkg-config \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    # Ubuntu 22.04's python3-opencv apt package is OpenCV 4.5.x
    # where cv2.aruco still has drawMarker (not generateImageMarker).
    # generate_aruco_pngs.py uses the 4.7+ name, so install a newer
    # opencv-contrib-python via pip instead.
    && pip3 install --no-cache-dir 'opencv-contrib-python>=4.7'

# ── Parrot apt repo + Sphinx packages ──────────────────────────────────
# Per https://developer.parrot.com/docs/sphinx/installation.html the apt
# repo is signed by debian.parrot.com.gpg. We install parrot-sphinx plus
# both UE4 worlds we use (empty for SDC arena, sphx-tests for the
# populated test scene).
#
# Docker-build hardening:
#   - parrot-sphinx's preinst script exits non-zero in a containerised
#     build because it tries to start firmwared/systemd services that
#     don't exist during `docker build`. Install supporting bits +
#     policy-rc.d (returns 101 = "do not start any service during
#     install") + a noop systemctl so the maintainer scripts succeed.
#   - The actual systemd / udev wiring is recreated at runtime by
#     the entrypoint, so masking it during build is safe.
RUN curl --fail --silent --show-error --location \
        https://debian.parrot.com/gpg \
      | gpg --dearmor \
      | tee /usr/share/keyrings/debian.parrot.com.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/debian.parrot.com.gpg] \
        https://debian.parrot.com/ $(lsb_release -cs) main generic" \
      | tee /etc/apt/sources.list.d/debian.parrot.com.list \
    && apt-get update \
    # supporting libs the parrot maintainer scripts call into
    && apt-get install -y --no-install-recommends \
        systemd systemd-sysv dbus udev kmod debconf-utils \
    # policy-rc.d: any service start during install returns 101 (skip)
    && printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod +x /usr/sbin/policy-rc.d \
    # stub systemctl so daemon-reload / start commands no-op
    && dpkg-divert --local --rename --add /bin/systemctl \
    && printf '#!/bin/sh\necho "[stub systemctl] $*" >&2\nexit 0\n' > /bin/systemctl \
    && chmod +x /bin/systemctl \
    # ── License pre-acceptance ─────────────────────────────────────
    # parrot-sphinx's preinst computes md5sum of
    # /opt/parrot-sphinx/usr/share/sphinx/license/content and skips
    # the EULA prompt entirely if it matches a known hash
    # (da368dad2f2296800a3f985a4c8f2d03 in the 2.15.1 build). On a
    # FIRST install that file doesn't exist yet (the deb hasn't
    # unpacked), so the script falls through to a debconf prompt
    # which can't be answered in a headless build.
    #
    # Trick: extract the license file from the deb's data archive
    # into the right path BEFORE apt-get install runs. preinst then
    # sees the matching md5, prints "License has not changed", and
    # skips the prompt.
    && apt-get download parrot-sphinx \
    && PSDEB=$(ls parrot-sphinx_*.deb | head -1) \
    && mkdir -p /tmp/ps-extract \
    && dpkg-deb -x "$PSDEB" /tmp/ps-extract \
    && mkdir -p /opt/parrot-sphinx/usr/share/sphinx/license \
    && cp /tmp/ps-extract/opt/parrot-sphinx/usr/share/sphinx/license/content \
          /opt/parrot-sphinx/usr/share/sphinx/license/content \
    && echo "── pre-seeded license file md5 ──" \
    && md5sum /opt/parrot-sphinx/usr/share/sphinx/license/content \
    # parrot install — DKMS modules (ulogger-dkms) may fail to build
    # against a non-existent kernel, that's expected and harmless
    # for sim use; force the install to continue past those.
    && (DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none \
            apt-get install -y --no-install-recommends \
                parrot-sphinx parrot-ue4-empty parrot-ue4-sphx-tests \
        || (echo "[parrot install] retrying with --fix-broken" \
            && dpkg --configure -a || true \
            && DEBIAN_FRONTEND=noninteractive \
                apt-get install -y --fix-broken --no-install-recommends \
                    parrot-sphinx parrot-ue4-empty parrot-ue4-sphx-tests)) \
    # restore real systemctl + drop policy-rc.d
    && rm -f /bin/systemctl \
    && dpkg-divert --local --rename --remove /bin/systemctl \
    && rm -f /usr/sbin/policy-rc.d \
    && rm -f parrot-sphinx_*.deb \
    && rm -rf /tmp/ps-extract \
    && rm -rf /var/lib/apt/lists/* \
    # Ensure the firmwared group exists (the postinst usually
    # creates it; we guard for the case it didn't run cleanly under
    # our license-md5 shortcut). Adding sdc to it happens AFTER
    # user creation, further down in the Dockerfile.
    && (getent group firmwared >/dev/null || groupadd firmwared)

# ── Tailscale ──────────────────────────────────────────────────────────
# Lets the operator reach sphinx-control + the camera stream from
# anywhere without exposing ports publicly. Inside a container we run
# tailscaled in `userspace-networking` mode (no /dev/net/tun needed).
RUN curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
        -o /usr/share/keyrings/tailscale-archive-keyring.gpg \
    && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list \
        -o /etc/apt/sources.list.d/tailscale.list \
    && apt-get update && apt-get install -y --no-install-recommends tailscale \
    && rm -rf /var/lib/apt/lists/*

# ── Blender (for tools/sphinx-arena/) ──────────────────────────────────
# build_full_arena.py + build_marker_fbx.py + build_painting_fbxs.py
# all need Blender 3.x in --background mode. Ubuntu 22.04 ships
# Blender 3.0.1 which matches what the SDC host uses.
RUN apt-get update && apt-get install -y --no-install-recommends blender \
    && rm -rf /var/lib/apt/lists/*

# ── User account ───────────────────────────────────────────────────────
# Sphinx refuses to run as root (its launcher script explicitly
# checks `id --name --user`). Create the same `sdc` user the
# production host uses so all paths line up.
ARG SDC_UID=1000
ARG SDC_GID=1000
RUN groupadd -g ${SDC_GID} sdc \
    && useradd -m -u ${SDC_UID} -g ${SDC_GID} -s /bin/bash sdc \
    && echo 'sdc ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers \
    && mkdir -p /home/sdc && chown -R sdc:sdc /home/sdc \
    # Add sdc to the firmwared group (created earlier in the parrot
    # install step) so `sphinx` can open /var/run/firmwared.sock.
    # Without this, drone spawn aborts with:
    #   "ERROR : You should be a member of the firmwared group to run sphinx."
    && usermod -aG firmwared sdc

# ── sdc-tobe repo ──────────────────────────────────────────────────────
# COPY from the build context rather than git-clone, because the
# upstream repo is private and the build sandbox has no credentials.
# The build context IS the repo (both `docker build .` from a local
# checkout AND the GitHub Actions workflow which runs after a
# checkout step), so this is also faster than cloning.
#
# The entrypoint can still `git pull` at runtime via SDC_GIT_PULL=1
# — that runs as user `sdc` who can supply credentials interactively
# or via a mounted ~/.git-credentials.
WORKDIR /home/sdc
COPY --chown=sdc:sdc . /home/sdc/sdc-tobe
USER sdc

# ── Python deps: TWO venvs ─────────────────────────────────────────────
# sphinx-control/.venv  — FastAPI/uvicorn dashboard (port 8090)
# .venv                 — flight-controller (controller_unified,
#                          parrot-olympe, etc.) — used when the
#                          operator hits "FC start" in the UI; path
#                          is hard-coded in sphinx-control/config.yaml
#                          as flight_controller.python.
# Both are pre-built so the entrypoint doesn't need network on every
# container start.
RUN python3 -m venv /home/sdc/sdc-tobe/sphinx-control/.venv \
    && /home/sdc/sdc-tobe/sphinx-control/.venv/bin/pip install --no-cache-dir \
        --upgrade pip \
    && if [ -f /home/sdc/sdc-tobe/sphinx-control/requirements.txt ]; then \
         /home/sdc/sdc-tobe/sphinx-control/.venv/bin/pip install --no-cache-dir \
            -r /home/sdc/sdc-tobe/sphinx-control/requirements.txt ; \
       else \
         /home/sdc/sdc-tobe/sphinx-control/.venv/bin/pip install --no-cache-dir \
            fastapi uvicorn pyyaml requests sqlalchemy psutil ; \
       fi \
    # Repo-level venv for the unified flight-controller. Its
    # requirements pull in parrot-olympe (large; amd64-only — that's
    # fine, the image targets linux/amd64). If FC startup still
    # fails for a missing module, add it to
    # controller_unified/requirements.txt and rebuild.
    && python3 -m venv /home/sdc/sdc-tobe/.venv \
    && /home/sdc/sdc-tobe/.venv/bin/pip install --no-cache-dir --upgrade pip wheel \
    && /home/sdc/sdc-tobe/.venv/bin/pip install --no-cache-dir \
        -r /home/sdc/sdc-tobe/controller_unified/requirements.txt

# ── Pre-build arena assets ─────────────────────────────────────────────
# Doing this at build time means a fresh container can serve the
# arena immediately — no first-launch Blender re-export.
#
# CROSS-BUILDING: Blender runs amd64-only and QEMU-emulating it on an
# M1 Mac (arm64 host) takes 30+ minutes for this step alone. Pass
# `--build-arg PREBUILD_ARENA=0` to skip; the entrypoint will build
# the arena on first container start instead. Default = 1 (build now).
ARG PREBUILD_ARENA=1
# Full pipeline:
#   make pngs        — generate ArUco marker PNGs (needs python3-opencv)
#   make fbxs        — Blender wraps each PNG into aruco_NNN.fbx
#   make arena-static— floor / wall / pillar / spectator FBXs
#   make logos       — fetch logo PNGs (best-effort; needs network)
#   make paintings   — Blender wraps logos into painting_*.fbx
#   make yaml        — emit arena.yml referencing all of the above
# `|| true` on logos so a network blip doesn't break the build —
# paintings just won't render (the YAML emitter skips missing FBXs).
RUN if [ "${PREBUILD_ARENA}" = "1" ]; then \
        set -e ; \
        cd /home/sdc/sdc-tobe/tools/sphinx-arena ; \
        make pngs ; \
        make fbxs ; \
        make arena-static ; \
        (make logos && make paintings) || echo "(logos/paintings skipped — non-fatal)" ; \
        make yaml ; \
    else \
        echo "PREBUILD_ARENA=0 — entrypoint will build on first start" ; \
    fi

# ── Entrypoint ─────────────────────────────────────────────────────────
USER root
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Ports
# 8090 — sphinx-control web UI / REST
# 9080 — drone-1 (sphinx) HTTP control endpoint
# 41251 — tailscale magicsock (UDP)
EXPOSE 8090 9080
EXPOSE 41251/udp

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sphinx-control"]
