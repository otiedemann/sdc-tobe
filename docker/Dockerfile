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
RUN curl --fail --silent --show-error --location \
        https://debian.parrot.com/gpg \
      | gpg --dearmor \
      | tee /usr/share/keyrings/debian.parrot.com.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/debian.parrot.com.gpg] \
        https://debian.parrot.com/ $(lsb_release -cs) main generic" \
      | tee /etc/apt/sources.list.d/debian.parrot.com.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        parrot-sphinx \
        parrot-ue4-empty \
        parrot-ue4-sphx-tests \
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
RUN cd /home/sdc/sdc-tobe/tools/sphinx-arena && make arena-static && make yaml \
    || echo "(arena pre-build best-effort; will rebuild at runtime if needed)"

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
