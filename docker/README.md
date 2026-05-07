# sdc-sphinx Docker image

Containerised SDC26 simulator: Ubuntu 22.04 + Parrot Sphinx 2.15 +
parrot-ue4-empty + parrot-ue4-sphx-tests + sphinx-control + Tailscale.

## Can Sphinx run truly headless?

**Short answer: no, but you don't need a real monitor.**

UE4 (the renderer Sphinx uses to produce the drone's camera feed)
requires a Vulkan-capable display surface. Without one it fails with
`InitSDL() failed`. The container side-steps this with **Xvfb** — a
virtual X server that gives UE4 a framebuffer to render into. The
NVIDIA Container Toolkit bind-mounts the GPU + Vulkan ICD into the
container at runtime, so Vulkan rendering still hits real hardware.

The drone's camera stream (and every Sphinx telemetry endpoint) is
served by Sphinx over its own ports — completely independent of the
UE4 viewport. **An operator never needs to see the UE4 window**; they
just hit `sphinx-control` over HTTP and consume the camera stream
through the drone's REST/RTSP endpoint.

If you ever DO want to peek at the UE4 viewport (debugging), `docker
exec` into the container and run `x11vnc -display :99` then point a
VNC client at the published port.

## Prerequisites

- Linux host with NVIDIA GPU (Pascal or newer, e.g. T4, A10, A40, RTX)
- NVIDIA driver ≥ 525 on the host
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

`docker info | grep Runtimes` should list `nvidia`.

## Build

```bash
cd /path/to/sdc-tobe
docker build -t sdc-sphinx:latest -f docker/Dockerfile .
```

Build args:
- `SDC_REPO_URL` (default: this repo on GitHub)
- `SDC_BRANCH`   (default: `main`)
- `BASE_IMAGE`   (default: `nvidia/cuda:12.2.2-runtime-ubuntu22.04`)

## Run locally

```bash
docker run --gpus all --rm -it \
    -p 8090:8090 -p 9080:9080 \
    -e TS_AUTHKEY=tskey-...    \
    -e TS_HOSTNAME=sdc-sphinx-docker \
    --cap-add NET_ADMIN --cap-add SYS_ADMIN \
    -v sphinx-state:/home/sdc/.parrot-sphinx \
    -v sphinx-logs:/home/sdc/sdc-tobe/sphinx-control/logs \
    sdc-sphinx:latest
```

Then open `http://localhost:8090/` (or `http://sdc-sphinx-docker.<your-tailnet>.ts.net:8090/`).

Or with compose:
```bash
export TS_AUTHKEY=tskey-...
docker compose -f docker/docker-compose.yml up --build
```

## Run on RunPod

1. **Create a Pod** with these settings:
   - **Container image**: `<your-registry>/sdc-sphinx:latest`
     (push the image: `docker push your-registry/sdc-sphinx:latest`).
     Or build from source via a custom template — RunPod supports
     pointing at a GitHub repo with a `Dockerfile` at the root, but
     since ours lives at `docker/Dockerfile` you'll need a custom
     build path.
   - **GPU**: any NVIDIA card with ≥ 8 GB VRAM — RTX A4000 / A10 /
     L4 are all fine. The Empty world is GPU-light; sphx-tests is
     heavier.
   - **Container disk**: ≥ 30 GB (parrot-ue4-sphx-tests alone is ~4 GB).
   - **Volume**: 20 GB at `/home/sdc/.parrot-sphinx` if you want flight
     logs to survive Pod restarts.
   - **Expose**: TCP `8090` (sphinx-control), TCP `9080` (drone API).
     RunPod will give you `https://<podid>-8090.proxy.runpod.net/`.
   - **Environment variables**:
     - `TS_AUTHKEY` = `tskey-...` — your Tailscale auth key (recommended).
     - `TS_HOSTNAME` = `runpod-sphinx` or similar.
     - `SDC_GIT_PULL` = `1` so each Pod restart pulls the latest code.

2. **Start the Pod.** First boot takes 30–60 s (Xvfb + tailscale + repo
   pull + sphinx-control bring-up).

3. **Reach it.** Two options:
   - **Tailscale**: open `http://<TS_HOSTNAME>.<your-tailnet>.ts.net:8090/`
     from any device on your tailnet.
   - **RunPod proxy**: open the auto-generated `https://...proxy.runpod.net/`
     URL from the Pod page.

4. **Spawn a drone** in sphinx-control. The drone's HTTP endpoint
   becomes available at `:9080` (or via Tailscale at the same port).

## Run on Hetzner / Lambda / any VM with root SSH

**RunPod and most "container-as-a-service" providers (Modal, Beam,
serverless Vast.ai, Salad) block the loop-mount and AppArmor calls
that firmwared needs.** The image won't work there. For RunPod
specifically the symptom is the drone goes STOPPED right after
spawn with `mount: Operation not permitted` in firmwared.log.

If you control the Docker daemon (DataCrunch, Lambda, Hetzner
Dedicated, your own box, on-demand Vast.ai), you can pass
`--privileged` and the image runs cleanly. One-shot bootstrap script
for fresh Ubuntu 22.04 GPU servers: [`cloud-bootstrap.sh`](cloud-bootstrap.sh).

Provider-specific walkthroughs:
- [DataCrunch](DATACRUNCH.md) — EU, hourly billing (~0.40 €/h RTX 4090)
- [Hetzner Dedicated](HETZNER.md) — DE, monthly billing (~80 €/Mt)

## Run on Lambda Labs / other GPU clouds

Same as RunPod — any host with NVIDIA Container Toolkit can run this.

```bash
docker pull <your-registry>/sdc-sphinx:latest
docker run --gpus all -d \
    --name sphinx \
    -p 8090:8090 -p 9080:9080 \
    -e TS_AUTHKEY=tskey-... \
    --cap-add NET_ADMIN --cap-add SYS_ADMIN \
    --restart unless-stopped \
    <your-registry>/sdc-sphinx:latest
```

## What's in the image

| Component | Source | Purpose |
|---|---|---|
| `parrot-sphinx` | apt: `debian.parrot.com` | Drone simulator core (gazebo) |
| `parrot-ue4-empty` | apt: `debian.parrot.com` | Minimal UE4 world (default for SDC arena) |
| `parrot-ue4-sphx-tests` | apt: `debian.parrot.com` | Test world (used by `sdc_arena_test`) |
| `xvfb` + `xdotool` | apt: `ubuntu` | Virtual display + automated F10 to hide HMI |
| `tailscale` | apt: `pkgs.tailscale.com` | Mesh VPN for remote access |
| `blender` | apt: `ubuntu` (3.0.1) | Builds arena FBXs |
| `sdc-tobe` repo | `git clone main` | sphinx-control + arena tooling |
| `sphinx-control venv` | pip | FastAPI app on port 8090 |
| Pre-built arena assets | `make arena-static && make yaml` | No first-launch Blender wait |

## Troubleshooting

**Container starts but `/api/system` 502s**
Check `docker logs <ctr>` — look for "starting sphinx-control on
0.0.0.0:8090 as user sdc". If you see Python tracebacks, the venv
likely missed a dependency; either rebuild with `--no-cache` or
`docker exec -it <ctr> bash` and `pip install` the missing module.

**Drone spawn fails: "Exit is pending..."**
Same stale-firmwared issue we hit on the bare-metal SDC host. Inside
the container: `docker exec -it <ctr> sudo systemctl restart firmwared`.
If firmwared isn't running as a service in the container, restart the
container — its boot scripts re-init firmwared.

**UE4 "InitSDL() failed"**
Xvfb didn't start. `cat /var/log/xvfb.log` inside the container.
Usually a missing X library — `apt install libxcb-xinerama0` etc.
Capture the error and report.

**Tailscale device shows offline**
`docker exec -it <ctr> tailscale status` — if "stopped", the auth key
expired or wasn't passed. Re-auth:
```bash
docker exec -it <ctr> tailscale --socket=/var/run/tailscale/tailscaled.sock \
    up --authkey=tskey-... --hostname=sdc-sphinx-docker
```

## Building on Apple Silicon (M1/M2/M3)

Yes — Docker buildx will cross-build a `linux/amd64` image on an
arm64 host. Two practical caveats:

- **It's slow.** apt installs run under QEMU emulation. Expect
  5–15 min for the base build, vs. ~3 min on native x86.
- **The pre-build arena step (Blender) is the slowest** under QEMU
  — 30+ min if not skipped. The default cross-build skips it
  (`PREBUILD_ARENA=0`) and lets the entrypoint build the arena on
  first container start (when it's running on real x86 hardware,
  ~30 s).

The `build-amd64.sh` helper does the right thing:

```bash
# Build for amd64, load locally (won't run on M1, but pushable)
./docker/build-amd64.sh

# Build + push directly to a registry
./docker/build-amd64.sh --push ghcr.io/yourname/sdc-sphinx:latest

# Force pre-build the arena anyway (slower, but every container start
# is instant after that)
./docker/build-amd64.sh --prebuild-arena
```

The image won't run usefully on the M1 itself — there's no Vulkan-
capable GPU passthrough — but you can pull and run it on any x86
GPU host (RunPod / Lambda / EC2 g4dn / your own box).

### Faster: build on a x86 GitHub Actions runner

`.github/workflows/docker-sphinx.yml` builds the image natively on a
GitHub-hosted ubuntu-22.04 runner (real x86) and pushes to GHCR.
Trigger it with:

```bash
gh workflow run docker-sphinx.yml
# or push a tag:
git tag docker-v0.1 && git push --tags
```

Pull from any host:
```bash
docker pull ghcr.io/<your-org>/sdc-sphinx:latest
```

This is the fastest option if you're on an M1 — no local QEMU work
at all, and the cached layers make subsequent builds quick.

## Files

```
docker/
├── README.md            ← this file
├── Dockerfile           ← image definition (PREBUILD_ARENA build-arg)
├── docker-compose.yml   ← local-host launcher
├── entrypoint.sh        ← Xvfb + tailscaled + lazy arena build + sphinx-control
├── build-amd64.sh       ← cross-build helper for M1/M2/M3 Macs
└── .dockerignore        ← keeps build context lean

.github/workflows/
└── docker-sphinx.yml    ← native amd64 build on GitHub Actions → GHCR
```
