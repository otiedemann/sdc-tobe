# Running sdc-sphinx on DataCrunch (EU, hourly billing)

[DataCrunch](https://datacrunch.io/) is a Finland-based GPU cloud
that gives you real Linux VMs with root SSH (not container slots),
billed per hour. RTX 4090 around 0.40 €/h — currently the cheapest
EU option that works for our image.

This walks through deploying `sdc-sphinx` from zero. ~10 minutes
end-to-end (5 min sign-up + 5 min for the script).

## 1. Create a DataCrunch account

1. Go to <https://datacrunch.io/> → **Sign up** (top right).
2. Verify your email.
3. **Add a payment method** — Stripe (credit card) or SEPA. EUR billing.
4. *(Optional)* Add a credit balance — e.g. €10 lets you run an RTX 4090 for ~25 hours.

## 2. Add an SSH key

You'll SSH into the server as `root`, so put your public key in your account:

1. Console → top-right user menu → **Account** → **SSH keys**
2. Paste your `~/.ssh/id_ed25519.pub` (or `~/.ssh/id_rsa.pub`)
3. Save

If you don't have a key yet:
```bash
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub   # paste this into DataCrunch
```

## 3. Deploy a server

Console → **Deploy** (top nav).

| Field | Value |
|---|---|
| **Instance type** | **1× RTX 4090** (or **1× RTX A6000** if you want more VRAM) |
| **Location** | **Finland (FIN-01)** — only DC for now |
| **OS image** | **Ubuntu 22.04** |
| **Storage** | **40 GB** (image is ~6 GB, plus telemetry logs) |
| **SSH key** | the key you uploaded |
| **Hostname** | `sphinx` (or anything) |

Click **Deploy**. The server is up in 30–60 seconds; you'll see a public IP.

## 4. SSH in and run the bootstrap

In your terminal:

```bash
ssh root@<public-ip>
```

You should land at a `root@sphinx:~#` prompt. Now run the bootstrap:

```bash
# Set your secrets first (no spaces around =)
export GHCR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx        # GitHub PAT, read:packages
export TS_AUTHKEY=tskey-auth-xxxxxxxxxxxxxxxxxxxxxxxxx   # optional, recommended
export TS_HOSTNAME=sphinx-dc                              # how it appears on Tailscale

# One-shot install + container launch
curl -sSL https://raw.githubusercontent.com/otiedemann/sdc-tobe/main/docker/cloud-bootstrap.sh \
  | sudo -E bash -s -- ghcr.io/otiedemann/sdc-sphinx:latest
```

The script:

1. Verifies you're on Ubuntu, installs the NVIDIA driver if missing
2. Installs Docker + NVIDIA Container Toolkit
3. **Crashes if the GPU isn't reachable** (`docker run --gpus all nvidia-smi`)
4. Logs into ghcr.io with your PAT
5. `docker pull` the image (~5 min the first time, image is ~6 GB)
6. Writes `/etc/systemd/system/sdc-sphinx.service` with `--privileged --gpus all`
7. Starts it
8. Waits for `/api/system` to respond on port 8090
9. Prints the dashboard URL

If you see **"GPU not reachable — aborting"** right after the driver step:

```bash
reboot
# wait ~20 s, then SSH back in
ssh root@<public-ip>
# re-run the same curl|bash command
```

The driver install becomes a no-op the second time and the rest finishes.

## 5. Reach the dashboard

Two ways depending on how you set up:

**With Tailscale** (recommended — private, persistent URL):
```
http://sphinx-dc.<your-tailnet>.ts.net:8090/
```

**Direct via DataCrunch's public IP**:
```
http://<public-ip>:8090/
```
You may need to open port 8090 in DataCrunch's firewall: console → server → Networking → allow inbound tcp/8090, tcp/9080, tcp/9081. Or skip this and only use Tailscale (no inbound holes).

## Day-to-day

| Action | Command |
|---|---|
| **Logs** | `journalctl -u sdc-sphinx -f` |
| **Restart** | `systemctl restart sdc-sphinx` |
| **Update to newer image** | `docker pull ghcr.io/otiedemann/sdc-sphinx:latest && systemctl restart sdc-sphinx` |
| **Stop** | `systemctl stop sdc-sphinx` |
| **Service state** | `systemctl status sdc-sphinx` |

## Save money when not flying

DataCrunch only bills you for **running** instances. To stop billing:

1. Console → your server → **Power off** (or destroy it entirely).
2. Powered-off servers cost only ~5 €/Mt for the disk.
3. Power on later → server boots, systemd starts the container, sphinx-control answers in ~30 s.

If you destroy the server entirely, just deploy a new one and re-run the same bootstrap. State is in Docker volumes which die with the server — but our flight logs sync via Tailscale anyway, so this is usually fine.

## Why DataCrunch over RunPod

RunPod is **container-as-a-service**: their hypervisor runs Docker, you give it an image. They block `--privileged` for multi-tenant security. Sphinx + firmwared need privileged for loop-mounting the simulated drone firmware → drone spawns die instantly.

DataCrunch gives you a **VM with root SSH**: you install Docker, you control its config, `--privileged` is your call. The image runs cleanly.

## Pricing reference (as of 2026)

| Instance | GPU | Hourly | Notes |
|---|---|---|---|
| 1× RTX 4090 | 24 GB | ~0.40 €/h | overkill for our image, but cheapest |
| 1× RTX A6000 | 48 GB | ~0.80 €/h | more headroom |
| 1× A100 40GB | 40 GB | ~1.10 €/h | enterprise-grade |
| 1× H100 80GB | 80 GB | ~2.50 €/h | massive overkill |

For sdc-sphinx, the **RTX 4090** is the right call — 24 GB VRAM is way more than UE4 + sphinx needs.

## If something fails

Paste the bootstrap output and the last 50 lines of `journalctl -u sdc-sphinx`. Most failures fall into one of three buckets:

| Symptom | Cause | Fix |
|---|---|---|
| "GPU not reachable" after driver install | Kernel module not loaded | `reboot`, re-run script |
| Container starts then exits | Image pull failed mid-way | `docker pull` manually with verbose, retry |
| Tailscale shows "stopped" | Auth key expired / single-use | Generate new key with **Reusable + Ephemeral**, restart container |
