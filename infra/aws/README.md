# infra/aws — host config for the sphinx AWS instance

Everything system-level that needs to live on the EC2 box, kept in git
so a rebuild from a fresh AMI (or migration to a new instance type)
restores the same setup with one command.

This directory mirrors the absolute paths it installs to, so the
install script is just `cp -a etc/. /etc/` plus `cp -a opt/. /opt/`.

## Contents

```
infra/aws/
├── install.sh                          # one-shot applier (idempotent)
├── etc/
│   ├── cron.d/
│   │   └── nightly-shutdown            # 21:00 Europe/Berlin daily stop
│   ├── X11/
│   │   └── xorg.conf.nvidia-headless   # NVIDIA Xorg on :99 (not Xvfb)
│   └── systemd/system/
│       ├── auto-shutdown.timer         # 3 h after boot, halt
│       ├── auto-shutdown.service       #   (target of the timer)
│       ├── xvfb.service                # Xorg on :99 (name is historical)
│       ├── sphinx-bootstrap.service    # oneshot: env + drone + FC
│       ├── marker-mission.service      # :8080 (combined app — Conflicts= sdc-fc.service)
│       ├── marker-mission.service.d/
│       │   └── wait-bootstrap.conf     # wait for sphinx-bootstrap
│       ├── fc-healthcheck.service      # restart FC if disconnected
│       ├── fc-healthcheck.timer        # fires every 60 s
│       └── c2-controller.service       # :8070 remote_web_controller
└── opt/sdc-tobe/sphinx-control/
    └── sphinx-bootstrap.sh             # the script run by sphinx-bootstrap.service
```

NOT in this directory but on the same host (because they come from the
parrot-sphinx deb package, not from this repo):

- `/etc/systemd/system/firmwared.service` — provided by parrot-sphinx
- `/etc/systemd/system/sphinx-control.service` — installed by `sphinx-control/install.sh`
  using `sphinx-control/systemd/sphinx-control.service` from this repo

## Boot order

```
firmwared.service ──┬── xvfb.service (NVIDIA Xorg on :99)
                    │
                    └── sphinx-control.service (:8090)
                            │
                            ▼
                     sphinx-bootstrap.service  (oneshot, ~60 s)
                            │   POST /api/environment
                            │   POST /api/drones
                            │   POST /api/fc
                            ▼
                     marker-mission.service (:8080)
                     c2-controller.service  (:8070)

(separate, fires once 3h after boot)  auto-shutdown.timer
```

## Shutdown policy

Two independent stops for safety:

| Mechanism                       | Fires when                            | Why |
|---------------------------------|---------------------------------------|-----|
| `auto-shutdown.timer`           | 3 hours after each boot               | caps any session at 3 h regardless of operator action |
| `/etc/cron.d/nightly-shutdown`  | 21:00 Europe/Berlin daily             | catches the case "operator booted late and might forget" |

Whichever fires first wins. Both end with `shutdown -h +1 "<reason>"` —
the EC2 instance must have **Shutdown behavior = Stop** (the default for
on-demand) so the OS halt translates to a stop, not a terminate.

### Override for a longer session

```bash
sudo systemctl stop auto-shutdown.timer      # cancel just this boot
sudo touch /var/run/no-auto-shutdown         # inhibit via path-condition
sudo systemctl disable --now auto-shutdown.timer   # permanent
sudo rm /etc/cron.d/nightly-shutdown          # disable 21:00 cron too
```

### What about starting back up?

Nothing on the instance can start itself once it's stopped. Three options:

1. Manual — AWS console or `aws ec2 start-instances --instance-id <id>`
2. Cron on another always-on box that runs `aws ec2 start-instances`
3. EventBridge Scheduler — see the recipe in the top-level README.md

## How to apply on a freshly-launched instance

```bash
cd /opt/sdc-tobe/infra/aws
sudo ./install.sh
```

The script copies the trees, runs `systemctl daemon-reload`, enables
units, and sets the host TZ to `Europe/Berlin` if it isn't already.
After this the next reboot will bring up everything in order.

## How to verify the apply worked

```bash
systemctl list-units --type=service --state=running | grep -E \
  'firmwared|xvfb|sphinx-control|marker-mission|c2-controller'
systemctl list-timers --no-pager | grep auto-shutdown
sudo ss -lntp | awk '/:8070|:8080|:8090|:8383/ {print $4}'
```

You should see all five units active, `auto-shutdown.timer` in the
timer list with a `Trigger` 3 h ahead of boot, and four ports listening
(plus `:8383` once UE4 has finished booting, ~60 s after sphinx-bootstrap
fires).

## Editing these files

Edit in the repo, commit, push, then on the AWS box:

```bash
cd /opt/sdc-tobe && git pull
sudo /opt/sdc-tobe/infra/aws/install.sh
# restart the services that changed:
sudo systemctl restart <unit>
```

The install.sh applies file changes; only a deliberate restart of a
service makes a unit's new `ExecStart` / Environment take effect — that
way you don't accidentally bounce the live FC just by editing a comment
in an unrelated file.
