# Running sdc-sphinx on AWS EC2 (Frankfurt, hourly)

The cheapest reliable hourly path that actually works for our image:
**g4dn.xlarge Spot in eu-central-1 (Frankfurt)** at ~€0.18/h, plus ~€3/Mt
EBS storage. For ad-hoc flying sessions of 30 min – 2 h this comes
out to under €1 per session.

The AWS quirks compared to DataCrunch / Hetzner: needs a one-time GPU
quota increase, security groups instead of just opening ports, and the
default user is `ubuntu` (sudo to root for the bootstrap).

## TL;DR for someone who just wants to fly

1. [Bump the GPU vCPU quota](#1-one-time-quota-increase) (~10 min, automated for small bumps)
2. [Launch a g4dn.xlarge Spot in eu-central-1 with Ubuntu 22.04](#2-launch-the-instance)
3. SSH in, run the bootstrap (~10 min for image pull + first-boot steps)
4. Open the dashboard via Tailscale and fly
5. **Stop** the instance when done — billing pauses, only ~€0.10/day for storage

```bash
ssh ubuntu@<public-ip>
sudo -i
export GHCR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxx
export TS_AUTHKEY=tskey-auth-xxxxxxxxxxxxxxxxxxxxx
export TS_HOSTNAME=sphinx-aws
curl -sSL https://raw.githubusercontent.com/otiedemann/sdc-tobe/main/docker/cloud-bootstrap.sh \
  | bash -s -- ghcr.io/otiedemann/sdc-sphinx:latest
```

## 1. One-time quota increase

New AWS accounts can launch 0 GPU vCPUs by default. You need to bump it.

1. Open <https://eu-central-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas>
   *(Note: quotas are **per-region** — make sure the top-right region selector is "Europe (Frankfurt) eu-central-1".)*
2. In the search box, paste:
   ```
   All G and VT Spot Instance Requests
   ```
3. Click the matching row → **Request quota increase**
4. New quota value: **4** (g4dn.xlarge has 4 vCPUs; if you also want Spot fallback to On-Demand, also bump *"Running On-Demand G and VT instances"* to 4)
5. Submit

Approval is automated for small amounts (≤16 vCPUs) and lands in 5–30 minutes. Higher needs a real human, ~24h.

While waiting, you can do everything else (key pair, security group) so you're ready to launch the moment the quota lands.

## 2. Create an SSH key in AWS

If you already have an SSH key pair in EC2 from previous work, skip this. Otherwise:

1. EC2 console (`eu-central-1` selected) → **Key Pairs** (left sidebar) → **Create key pair**
2. Name: `sdc-sphinx`
3. Type: **ED25519**
4. Format: **.pem** (Mac/Linux) or **.ppk** (Windows + PuTTY)
5. **Create**. Browser downloads `sdc-sphinx.pem`. Move to `~/.ssh/` and chmod:
   ```bash
   mv ~/Downloads/sdc-sphinx.pem ~/.ssh/
   chmod 600 ~/.ssh/sdc-sphinx.pem
   ```

Or — easier if you already have a personal key — **Import key pair**: paste your existing `~/.ssh/id_ed25519.pub`. Then you don't need a new `.pem`, just use your existing `~/.ssh/id_ed25519`.

## 3. Create a security group

Restricts which IPs can hit which ports on your instance.

1. EC2 console → **Security Groups** → **Create security group**
2. Name: `sdc-sphinx-sg`
3. Description: `SSH for sphinx; everything else via Tailscale`
4. VPC: leave default
5. **Inbound rules**: add one
   - Type: **SSH**
   - Source: **My IP** (auto-fills your current public IP)
6. **Outbound rules**: leave default (allow all)
7. **Create**

That's it. We **deliberately do NOT open** TCP 8090/9080/9081 publicly — operators reach the dashboard via Tailscale, not over the public internet. If your home IP changes, edit this rule's source to the new "My IP".

If you want public dashboard access too (not recommended): add inbound TCP 8090, 9080, 9081 with source = "My IP" or "Anywhere" (the latter exposes you to scanners — only do this if there's a real reason).

## 4. Launch the instance

1. EC2 console → **Instances** → **Launch instances**
2. **Name**: `sphinx-aws`
3. **Application and OS Images**: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**, **64-bit (x86)**.
4. **Instance type**: type `g4dn.xlarge` in the box. Verify it shows: 4 vCPU, 16 GiB RAM, 1× T4.
5. **Key pair**: select the one from step 2 (`sdc-sphinx` or your imported key)
6. **Network settings**:
   - **Existing security group** → select `sdc-sphinx-sg`
7. **Configure storage**:
   - **40 GiB**, **gp3** (faster + cheaper than gp2)
8. **Advanced details** → scroll down to:
   - **Purchasing option**: ✓ **Request Spot instances**
   - **Maximum price** (under the Spot toggle): leave blank (defaults to On-Demand price = $0.526). Setting a max ≥ On-Demand means you essentially never get outbid, while still benefiting from the Spot discount.
   - **Interruption behavior**: **Stop** (preserves the EBS volume; you can restart on the same disk on another Spot instance)
9. Leave the rest at defaults.
10. **Launch instance**.

Wait ~30–60 s for the instance to boot. The Public IP appears on the instance's row in the **Instances** list.

## 5. SSH in and bootstrap

From your laptop:

```bash
# Substitute your key path if different
ssh -i ~/.ssh/sdc-sphinx.pem ubuntu@<public-ip>
```

You'll land at `ubuntu@ip-172-31-x-x:~$`. Switch to root for the bootstrap (the script needs to install packages, set up systemd, etc.):

```bash
sudo -i
```

Now run the bootstrap, with your secrets:

```bash
export GHCR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxx          # GitHub PAT, read:packages
export TS_AUTHKEY=tskey-auth-xxxxxxxxxxxxxxxxxxxxx     # Tailscale, reusable+ephemeral
export TS_HOSTNAME=sphinx-aws

curl -sSL https://raw.githubusercontent.com/otiedemann/sdc-tobe/main/docker/cloud-bootstrap.sh \
  | bash -s -- ghcr.io/otiedemann/sdc-sphinx:latest
```

The script will:
1. Install NVIDIA driver (~3 min) — first-time only
2. Install Docker + NVIDIA Container Toolkit
3. Verify `docker run --gpus all` reaches the GPU (fails fast if not)
4. Log into ghcr.io with your PAT
5. `docker pull` the image (~5 min, first time, ~6 GB)
6. Write `/etc/systemd/system/sdc-sphinx.service`
7. Start it, wait for `/api/system` to respond, print the URL

If the script complains "GPU not reachable — aborting" right after the driver install, **reboot the instance** (`reboot` in the SSH session, then SSH back in) and re-run the same `curl ... | bash` line. Driver install becomes a no-op the second time.

## 6. Reach the dashboard

Once the script finishes, two ways to reach it:

**Via Tailscale** (recommended):
```
http://sphinx-aws.<your-tailnet>.ts.net:8090/
```

**Via public IP** (only if you opened 8090 in the security group):
```
http://<aws-public-ip>:8090/
```

Inside the dashboard:
1. **Start environment** (default: `sdc_arena`, `1280×1024`, panels hidden)
2. Wait for green "running" pill
3. **Spawn drone** (only the `anafi-4k` profile is available, which uses the pre-baked firmware)
4. Drone goes to RUNNING and stays
5. **Start FC** with Anafi IP `127.0.0.1`

Click around, fly, do whatever you need.

## 7. Stop the instance to save money

When you're done flying, **stop** (don't terminate) the instance:

In the EC2 console:
- Select the instance row
- **Instance state** → **Stop instance** → confirm

Or via CLI:
```bash
aws ec2 stop-instances --region eu-central-1 --instance-ids i-xxxxxxxxxxxxx
```

Stopped Spot instances **don't bill compute**. You only pay:
- EBS storage: ~$0.08/GB/Mt for gp3 → 40 GB = **$3.20/Mt** = ~€0.10/day
- Public IP (Elastic IP if assigned): $0.005/h while *not attached* — none at default

So an idle stopped instance costs ~€3/Mt to keep around. Most operators just stop, fly later, stop, etc.

To resume:
- EC2 console → select instance → **Instance state** → **Start instance**
- Or `aws ec2 start-instances --region eu-central-1 --instance-ids i-...`
- Boot takes 1–2 min, then systemd auto-starts the container, dashboard answers in ~30 s
- **The Public IP changes on every start** (Spot doesn't preserve EIP). Tailscale handles this — `sphinx-aws.<tailnet>.ts.net` resolves to whatever the current IP is, you don't need to re-bookmark.

When you're really done with this whole project: **Terminate** the instance (deletes the EBS volume too). Costs €0 from then on.

## 8. Spot interruption handling

Spot instances can be **reclaimed** by AWS with 2 minutes notice if capacity gets tight. For g4dn in eu-central-1 this is rare (T4 inventory is huge, demand has shifted to A10/L4) — I run g4dn Spot for weeks without seeing one. But it can happen.

If it does:
- Your instance gets a `state=terminated` notification 2 min before shutdown
- The "Interruption behavior: Stop" you selected at launch means the EBS volume is preserved
- AWS doesn't auto-launch a replacement — you launch a new instance manually and attach the old volume to it

The image's state lives in Docker volumes which die with the instance. Important state to back up before/after each session:
```bash
# In SSH on the instance:
docker run --rm \
    -v sphinx-state:/source -v /tmp:/dest \
    alpine tar -czf /dest/sphinx-state-$(date +%Y%m%d-%H%M).tar.gz -C /source .
# Then scp /tmp/sphinx-state-*.tar.gz to your laptop
```

For SDC's "occasional flying" pattern this is overkill — a Spot reclaim mid-session means you re-run the bootstrap on a fresh instance and resume in ~10 minutes. State you really care about (flight logs in `/home/sdc/sdc-tobe/sphinx-control/logs`) was probably already pulled to your laptop via Tailscale anyway.

If you absolutely cannot tolerate interruption (e.g. competition day): launch the same g4dn.xlarge as **On-Demand** instead of Spot — costs $0.53/h instead of $0.18/h, no interruption risk.

## 9. Cost worked example

Typical SDC dev session: **2 hours** of g4dn.xlarge Spot.

| Component | Rate | This session |
|---|---|---|
| Compute (Spot) | ~$0.18/h | $0.36 |
| EBS (40 GB gp3) | ~$3.20/Mt = ~$0.004/h | $0.01 |
| Egress to Tailscale | first 100 GB/Mt free | $0.00 |
| **Total** | | **~$0.37 ≈ €0.34** |

Compared to:
- DataCrunch RTX 4090 (when available): ~€0.40 × 2 h = €0.80
- Verda A100 80GB: ~€1.20 × 2 h = €2.40
- Hetzner Dedicated AX42 monthly: €80/Mt = €2.67/day even if you don't fly that day

For ad-hoc flying, AWS Spot wins by 2-3×. For 24/7 use, Hetzner Dedicated wins.

## 10. Daily ops cheatsheet

```bash
# Service control (in SSH on the instance)
systemctl status   sdc-sphinx
systemctl restart  sdc-sphinx
systemctl stop     sdc-sphinx
journalctl -u      sdc-sphinx -f

# Update to a newer image
docker pull ghcr.io/otiedemann/sdc-sphinx:latest
systemctl restart sdc-sphinx

# Tear down completely
systemctl disable --now sdc-sphinx
docker rm -f sdc-sphinx
docker volume rm sphinx-state sphinx-logs   # only if you don't need the state
```

```bash
# Instance lifecycle (from your laptop, via aws CLI)
aws ec2 describe-instances --region eu-central-1 \
    --filters "Name=tag:Name,Values=sphinx-aws" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' \
    --output table

aws ec2 stop-instances  --region eu-central-1 --instance-ids i-xxxxxxxxxxxxx
aws ec2 start-instances --region eu-central-1 --instance-ids i-xxxxxxxxxxxxx
```

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Quota increase pending >24h | Hit auto-approval limit (>16 vCPUs) | Open AWS support case asking for the bump; usually responded within 4 h |
| `Insufficient capacity` when launching Spot | g4dn temporarily oversubscribed in eu-central-1 | Try eu-west-1 (Ireland), or fall back to On-Demand for one session |
| `nvidia-smi` fails after driver install | Kernel module not loaded | `reboot`, retry the bootstrap (driver install becomes a no-op) |
| sphinx-control 502 right after bootstrap | Service still warming up | wait 30 s; if persists, `journalctl -u sdc-sphinx -f` for actual error |
| Tailscale shows the device but URL 404s | sphinx-control crashed inside the container | `docker logs sdc-sphinx` for traceback |
| Spot instance terminated unexpectedly | Reclaim — sudden | Launch a new one. Attach old EBS volume if state matters. |
| EBS volume runs out of space | Container logs accumulating | `docker volume prune`, or expand the volume in the EC2 console |

## 12. Going further: AWS CLI scripted launch

If you'll launch + tear down often, scripting is faster than the console. After installing the AWS CLI and `aws configure`-ing your credentials:

```bash
# One-shot launch (Spot, 40 GB disk, our security group, our key)
aws ec2 run-instances \
    --region eu-central-1 \
    --image-id ami-0a628e1e89aaedf80 \
    --instance-type g4dn.xlarge \
    --instance-market-options 'MarketType=spot,SpotOptions={InstanceInterruptionBehavior=stop,SpotInstanceType=persistent}' \
    --key-name sdc-sphinx \
    --security-groups sdc-sphinx-sg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sphinx-aws}]' \
    --count 1
```

(The AMI ID `ami-0a628e1e89aaedf80` is Ubuntu 22.04 in eu-central-1 as of 2026 — may have rolled forward; query the latest with `aws ec2 describe-images --region eu-central-1 --owners 099720109477 --filters 'Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*' --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text`.)

I won't bake this into a script unless you really want it — manual console + bootstrap is reasonable for ad-hoc use.
