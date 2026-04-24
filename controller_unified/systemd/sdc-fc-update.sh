#!/usr/bin/env bash
# Boot-time sync: stash any local changes and force the working tree to
# origin/<branch>. Runs once via sdc-fc-update.service before the flight
# controller starts. Must stay resilient: if the network is down or github
# is unreachable we still exit 0 so the flight controller starts with
# whatever revision is on disk.

set -u
set -o pipefail

REPO="${SDC_FC_REPO:-/home/sdc/sdc-tobe}"
BRANCH="${SDC_FC_BRANCH:-main}"

log() { echo "[sdc-fc-update] $*"; }

if [[ ! -d "$REPO/.git" ]]; then
    log "no git repo at $REPO — nothing to sync"
    exit 0
fi

cd "$REPO"

# Non-interactive SSH: never prompt for passphrase / host-key / password.
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"
export GIT_TERMINAL_PROMPT=0

# Probe origin first with a short timeout so a flaky link doesn't wedge boot.
if ! timeout 15 git ls-remote --exit-code origin "$BRANCH" >/dev/null 2>&1; then
    log "origin unreachable — keeping current revision ($(git log -1 --oneline 2>/dev/null || echo 'unknown'))"
    exit 0
fi

# Snapshot HEAD before we touch anything so a developer can always recover.
PRE_REF="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$PRE_REF" ]]; then
    git update-ref "refs/sdc-fc/pre-pull/$(date +%Y%m%d-%H%M%S)" "$PRE_REF" || true
fi

# Stash local changes (tracked + untracked) if any.
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    STASH_MSG="sdc-fc-update auto-stash $(date -Iseconds)"
    log "local changes detected — stashing as: $STASH_MSG"
    git stash push --include-untracked -m "$STASH_MSG" || log "stash failed — continuing anyway"
fi

# Pull + force-reset. We do NOT merge; the flight controller fleet should
# always run exactly what's on origin.
if ! timeout 30 git fetch --prune origin; then
    log "git fetch failed — keeping current revision"
    exit 0
fi
git checkout -B "$BRANCH" "origin/$BRANCH" >/dev/null 2>&1 || true
git reset --hard "origin/$BRANCH"

# Keep the venv up to date if one exists. Non-fatal: a failed install
# shouldn't stop the flight controller from starting.
if [[ -x "$REPO/.venv/bin/pip" && -f "$REPO/controller_unified/requirements.txt" ]]; then
    log "refreshing venv dependencies"
    "$REPO/.venv/bin/pip" install --quiet --disable-pip-version-check \
        -r "$REPO/controller_unified/requirements.txt" \
        || log "pip install failed — continuing"
fi

log "now at $(git log -1 --oneline)"
exit 0
