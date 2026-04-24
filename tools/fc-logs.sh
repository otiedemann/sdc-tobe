#!/usr/bin/env bash
# Live-tail sdc-fc.service logs from every flight controller in a single
# tmux window — one pane per host, pane border shows hostname + live
# connection state. Auto-reconnects with backoff when a host drops,
# restarts, or reboots.
#
# Usage:
#   /tmp/fc-logs.sh                              # auto-discover flightctrl1..8
#   /tmp/fc-logs.sh flightctrl4 flightctrl5      # explicit host list
#   FC_HOSTS="ctrl4 ctrl5" /tmp/fc-logs.sh       # via env var
#
# tmux cheatsheet (prefix = Ctrl-b):
#   Ctrl-b d           detach (leaves SSH/journalctl running in background)
#   Ctrl-b o           rotate focus through panes
#   Ctrl-b z           zoom current pane
#   Ctrl-b :kill-ses   close every pane and disconnect

set -u

if ! command -v tmux >/dev/null; then
    echo "tmux is required — install with: sudo apt install tmux" >&2
    exit 1
fi

# Host list: CLI args > $FC_HOSTS > auto-discover flightctrl1..8
if [[ $# -gt 0 ]]; then
    HOSTS=("$@")
elif [[ -n "${FC_HOSTS:-}" ]]; then
    # shellcheck disable=SC2206
    HOSTS=(${FC_HOSTS})
else
    echo "auto-discovering flight controllers…"
    HOSTS=()
    for n in 1 2 3 4 5 6 7 8; do
        h="flightctrl$n"
        if timeout 2 ssh -o BatchMode=yes -o ConnectTimeout=1 -o StrictHostKeyChecking=accept-new \
                sdc@"$h" true 2>/dev/null; then
            HOSTS+=("$h")
            echo "  ✓ $h"
        fi
    done
fi

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "No flight controllers reachable. Pass hostnames explicitly:" >&2
    echo "    $0 flightctrl4 flightctrl5" >&2
    exit 1
fi

# ── Pane runner: one process per host, owned by its tmux pane.
# Follows the journal over SSH and auto-reconnects with exponential
# backoff (1s → 30s cap). Updates its own pane title so the border
# shows live connection state: ✓ while a session is active, ⚠ while
# retrying. Resets backoff to 1s once a session has lasted >60s
# (distinguishes "bad network" from "transient disconnect").
PANE_RUNNER="$(mktemp --tmpdir fc-logs-pane.XXXXXX.sh)"
cat > "$PANE_RUNNER" <<'PANE_EOF'
#!/usr/bin/env bash
set -u
h="${1:?hostname required}"

set_title() {
    # TMUX_PANE is set by tmux; outside tmux this is a no-op.
    [[ -n "${TMUX_PANE:-}" ]] && tmux select-pane -t "$TMUX_PANE" -T "$1" 2>/dev/null || true
}

trap 'set_title "$h (exit)"; exit 0' INT TERM

initial=true
delay=1
attempts=0
while :; do
    if [[ "$initial" == true ]]; then
        printf '\033[1;36m── connecting to %s ──\033[0m\n' "$h"
        journal_args=(-n 30 -f)
        initial=false
    else
        attempts=$((attempts + 1))
        printf '\033[1;33m── reconnecting to %s (attempt #%d) ──\033[0m\n' "$h" "$attempts"
        # Skip backlog on reconnect so we don't re-show lines that were
        # already in the pane before the drop.
        journal_args=(-f)
    fi

    set_title "$h ✓"
    start=$SECONDS
    ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
        -o ConnectTimeout=5 -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        sdc@"$h" "journalctl -u sdc-fc.service ${journal_args[*]} --no-hostname --output=short-precise"
    rc=$?
    elapsed=$((SECONDS - start))
    set_title "$h ⚠ reconn"

    # A session that lasted >60s was "healthy" — the disconnect is
    # probably a transient blip (WiFi glitch, unit restart). Reset the
    # backoff so we come back fast instead of waiting 30s.
    if (( elapsed > 60 )); then
        delay=1
        attempts=0
    fi

    printf '\033[1;31m── %s: disconnected (rc=%d, session %ds) — retrying in %ds ──\033[0m\n' \
        "$h" "$rc" "$elapsed" "$delay"
    sleep "$delay"
    # Exponential backoff, capped at 30s
    delay=$(( delay * 2 > 30 ? 30 : delay * 2 ))
done
PANE_EOF
chmod +x "$PANE_RUNNER"

SESSION="fclogs-$$"

# First pane
tmux new-session -d -s "$SESSION" -n logs "$PANE_RUNNER ${HOSTS[0]}"
tmux select-pane -t "$SESSION":logs -T "${HOSTS[0]}"

# Additional panes
for ((i=1; i<${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    tmux split-window -t "$SESSION":logs "$PANE_RUNNER $h"
    tmux select-pane -t "$SESSION":logs -T "$h"
    tmux select-layout -t "$SESSION":logs tiled >/dev/null
done

# Show live pane titles on each border
tmux set-option -t "$SESSION":logs pane-border-status top >/dev/null
tmux set-option -t "$SESSION":logs pane-border-format ' #{pane_title} ' >/dev/null

# Equalize and attach
tmux select-layout -t "$SESSION":logs tiled >/dev/null
tmux attach -t "$SESSION"

# After detach/exit: panes are already forked + have the runner script
# parsed into memory, so cleaning the tempfile is safe even if the user
# re-attaches later.
rm -f "$PANE_RUNNER"
