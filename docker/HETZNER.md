# Running sdc-sphinx on Hetzner (or any Linux VM with root)

Hetzner Cloud GPU + Hetzner Dedicated both work. So do Lambda Labs,
DataCrunch, OVHcloud, or any provider that gives you a Linux VM with
root SSH and an NVIDIA GPU.

The Hetzner option is the cheapest pay-as-you-go path in the EU. As
of 2026:

| Plan | Price | Use case |
|---|---|---|
| Hetzner Cloud **GEX44** (RTX 4000 Ada, 32 GB RAM) | ~0.91 €/h, capped at ~149 €/month | pay-per-hour, stop the server when not flying |
| Hetzner Dedicated **AX-** with consumer GPU | ~80 €/month fixed | always-on, cheapest if you fly daily |

## One-shot install

After a Hetzner Cloud GPU server with **Ubuntu 22.04** is up, SSH in
and run:

```bash
export GHCR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxx          # read:packages
export TS_AUTHKEY=tskey-auth-xxxxxxxxxxxxxxxxxxxxx    # optional but recommended
export TS_HOSTNAME=sphinx-hetzner

curl -sSL https://raw.githubusercontent.com/otiedemann/sdc-tobe/main/docker/hetzner-bootstrap.sh \
  | sudo -E bash -s -- ghcr.io/otiedemann/sdc-sphinx:latest
```

The `-E` preserves your env vars through `sudo`. The script:

1. Installs the NVIDIA driver (skip if already present)
2. Installs Docker + NVIDIA Container Toolkit
3. Verifies `docker run --gpus all` actually reaches the GPU
4. Logs into ghcr.io with your PAT
5. Pulls the image (~6 GB, takes ~5 min the first time)
6. Writes a systemd unit at `/etc/systemd/system/sdc-sphinx.service`
7. Starts the container with `--privileged --gpus all` plus
   port 8090 (sphinx-control), 9080/9081 (drone APIs)
8. Waits for `/api/system` to respond, then prints the URL

If the script complains *"GPU not reachable — aborting"* right after
installing the driver: that's the fresh kernel module not loaded.
**Reboot the server**, SSH back in, and re-run the same command —
the driver step is now a no-op and the rest finishes.

## What you get

- `systemctl status sdc-sphinx` — service state
- `journalctl -u sdc-sphinx -f` — entrypoint + container stdout
- `docker logs sdc-sphinx -f` — same, via Docker
- Reach the dashboard at `http://<server-ip>:8090/` (or via Tailscale at `http://sphinx-hetzner.<your-tailnet>.ts.net:8090/` if you set `TS_AUTHKEY`)

## Why not RunPod?

Sphinx + firmwared need `CAP_SYS_ADMIN` to loop-mount the simulated
drone's firmware ext2. RunPod and most "container-as-a-service" GPU
providers (Vast.ai serverless, Modal, Beam, Salad) block this for
security on shared multi-tenant hosts. **Even RunPod's "secure cloud"
template** doesn't expose privileged mode.

If you want the cheapest option that *also* allows `--privileged`:
**Vast.ai with on-demand instances** has it (~0.20 $/h for an RTX
3090) and the host machines are individually rented. Same setup
flow as Hetzner — just SSH in and run this script.

## Updating

When a new image is published to GHCR (via the `docker-sphinx.yml`
GitHub Actions workflow):

```bash
docker pull ghcr.io/otiedemann/sdc-sphinx:latest
sudo systemctl restart sdc-sphinx
```

The systemd unit's `--rm` flag means it always picks up the latest
pulled image on restart.

## Stopping / removing

```bash
sudo systemctl disable --now sdc-sphinx
sudo docker rm -f sdc-sphinx
sudo rm /etc/systemd/system/sdc-sphinx.service
```

State (telemetry logs, sphinx-control DB) lives in the Docker
volumes `sphinx-state` and `sphinx-logs`. Remove them too if you
want a fully clean slate:
```bash
sudo docker volume rm sphinx-state sphinx-logs
```
