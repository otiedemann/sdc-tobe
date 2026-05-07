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
        build-essential pkg-config \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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
    # parrot-sphinx's preinst exits with "License not approved.
    # Installation will abort." unless the EULA was accepted. Cover
    # all the mechanisms the maintainer scripts could be using:
    #   1. debconf-set-selections for the various plausible keys
    #   2. SPHINX_ACCEPT_EULA env var (common pattern)
    #   3. Pre-create marker files at the locations the script
    #      probably checks
    # We dump the actual preinst at the start so the build log
    # shows exactly what gates the install — useful if Parrot
    # changes the gate in a future release.
    && apt-get download parrot-sphinx \
    && PSDEB=$(ls parrot-sphinx_*.deb | head -1) \
    && mkdir -p /tmp/ps-control \
    && dpkg-deb -e "$PSDEB" /tmp/ps-control \
    && echo "── parrot-sphinx preinst (first 80 lines) ──" \
    && head -80 /tmp/ps-control/preinst 2>&1 || true \
    && echo "── parrot-sphinx templates ──" \
    && cat /tmp/ps-control/templates 2>&1 || true \
    && echo "── end ──" \
    && for q in license eula license-accepted accept-license accept-eula; do \
         echo "parrot-sphinx parrot-sphinx/$q boolean true" \
           | debconf-set-selections; \
         echo "parrot-sphinx parrot-sphinx/$q select Yes" \
           | debconf-set-selections; \
         echo "parrot-sphinx parrot-sphinx/$q seen true" \
           | debconf-set-selections; \
       done \
    # Common file markers some packages check
    && mkdir -p /etc/parrot-sphinx /var/lib/parrot-sphinx \
    && touch /etc/parrot-sphinx/license-accepted \
             /var/lib/parrot-sphinx/license-accepted \
             /etc/parrot-sphinx/.eula-accepted \
    # Env vars the preinst might honour
    && export SPHINX_ACCEPT_EULA=yes \
             SPHINX_LICENSE_ACCEPTED=yes \
             PARROT_LICENSE_ACCEPTED=yes \
             PARROT_SPHINX_LICENSE_ACCEPTED=yes \
             ACCEPT_EULA=yes \
    # parrot install — DKMS modules (ulogger-dkms) may fail to build
    # against a non-existent kernel, that's expected and harmless
    # for sim use; force the install to continue past those.
    #
    # The parrot-sphinx preinst calls `read` directly to ask the
    # operator to type "yes" for the EULA — debconf seeding alone
    # doesn't cover that. We pipe `yes yes` (which prints "yes\n"
    # forever) into apt-get's stdin so the preinst's read gets a
    # "yes" answer. APT_LISTCHANGES_FRONTEND=none + DEBIAN_FRONTEND
    # noninteractive prevent other prompts from blocking.
    && (yes yes | DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none \
            apt-get install -y --no-install-recommends \
                parrot-sphinx parrot-ue4-empty parrot-ue4-sphx-tests \
        || (echo "[parrot install] retrying with --fix-broken" \
            && yes yes | dpkg --configure -a || true \
            && yes yes | DEBIAN_FRONTEND=noninteractive \
                apt-get install -y --fix-broken --no-install-recommends \
                    parrot-sphinx parrot-ue4-empty parrot-ue4-sphx-tests)) \
    # restore real systemctl + drop policy-rc.d
    && rm -f /bin/systemctl \
    && dpkg-divert --local --rename --remove /bin/systemctl \
    && rm -f /usr/sbin/policy-rc.d \
    && rm -f parrot-sphinx_*.deb \
    && rm -rf /tmp/ps-control \
    && rm -rf /var/lib/apt/lists/*

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
    && mkdir -p /home/sdc && chown -R sdc:sdc /home/sdc

# ── sdc-tobe repo ──────────────────────────────────────────────────────
# Default branch is main. Override with `--build-arg SDC_BRANCH=...`.
ARG SDC_REPO_URL=https://github.com/otiedemann/sdc-tobe.git
ARG SDC_BRANCH=main
USER sdc
WORKDIR /home/sdc
RUN git clone --branch "${SDC_BRANCH}" --depth 50 \
        "${SDC_REPO_URL}" /home/sdc/sdc-tobe

# ── Python deps for sphinx-control ─────────────────────────────────────
# Build a project-local venv so the entrypoint doesn't need network on
# every container start. requirements.txt lives at the repo root.
RUN python3 -m venv /home/sdc/sdc-tobe/sphinx-control/.venv \
    && /home/sdc/sdc-tobe/sphinx-control/.venv/bin/pip install --no-cache-dir \
        --upgrade pip \
    && if [ -f /home/sdc/sdc-tobe/sphinx-control/requirements.txt ]; then \
         /home/sdc/sdc-tobe/sphinx-control/.venv/bin/pip install --no-cache-dir \
            -r /home/sdc/sdc-tobe/sphinx-control/requirements.txt ; \
       else \
         /home/sdc/sdc-tobe/sphinx-control/.venv/bin/pip install --no-cache-dir \
            fastapi uvicorn pyyaml requests sqlalchemy psutil ; \
       fi

# ── Pre-build arena assets ─────────────────────────────────────────────
# Doing this at build time means a fresh container can serve the
# arena immediately — no first-launch Blender re-export.
#
# CROSS-BUILDING: Blender runs amd64-only and QEMU-emulating it on an
# M1 Mac (arm64 host) takes 30+ minutes for this step alone. Pass
# `--build-arg PREBUILD_ARENA=0` to skip; the entrypoint will build
# the arena on first container start instead. Default = 1 (build now).
ARG PREBUILD_ARENA=1
RUN if [ "${PREBUILD_ARENA}" = "1" ]; then \
        cd /home/sdc/sdc-tobe/tools/sphinx-arena \
        && make arena-static && make yaml \
        || echo "(arena pre-build best-effort; will rebuild at runtime if needed)" ; \
    else \
        echo "PREBUILD_ARENA=0 — skipping arena build (entrypoint will run it on first start)" ; \
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
