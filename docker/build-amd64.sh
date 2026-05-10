#!/usr/bin/env bash
# Cross-build the sdc-sphinx image for linux/amd64 from any host
# (including Apple Silicon Macs).
#
# Usage:
#   ./docker/build-amd64.sh                          # build only, load locally
#   ./docker/build-amd64.sh --push ghcr.io/me/sdc-sphinx:latest
#
# Notes for M1/M2/M3 Mac users:
#   - The amd64 base image runs under QEMU emulation, so the build
#     is slow (5–15 min — the apt installs are CPU-bound emulation).
#   - PREBUILD_ARENA=0 is the default here so we skip the Blender
#     step (which is the slowest under QEMU). The entrypoint builds
#     the arena on first container start instead.
#   - The resulting image runs on x86 hosts only. You won't be able
#     to launch it on the M1 itself with GPU access.

set -euo pipefail

cd "$(dirname "$0")/.."

PLATFORM="${PLATFORM:-linux/amd64}"
TAG="${TAG:-sdc-sphinx:latest}"
PREBUILD_ARENA="${PREBUILD_ARENA:-0}"  # M1 default: skip slow Blender step
PUSH=""
EXTRA_ARGS=()

# Parse `--push <tag>` form
while [ $# -gt 0 ]; do
    case "$1" in
        --push)
            PUSH="--push"
            if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
                TAG="$2"; shift 2
            else
                shift
            fi
            ;;
        --tag|-t)
            TAG="$2"; shift 2 ;;
        --platform)
            PLATFORM="$2"; shift 2 ;;
        --prebuild-arena)
            PREBUILD_ARENA=1; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Ensure buildx is available + a builder exists.
if ! docker buildx version >/dev/null 2>&1; then
    echo "ERROR: docker buildx not available. Update Docker Desktop or install buildx." >&2
    exit 1
fi
if ! docker buildx inspect sdc-builder >/dev/null 2>&1; then
    echo "[build] creating buildx builder 'sdc-builder'"
    docker buildx create --name sdc-builder --use --driver docker-container
fi
docker buildx use sdc-builder

# When building locally (no --push), --load is required to import the
# image into the local Docker daemon. --load only works for single
# platforms, which is fine since we always target one.
LOAD_OR_PUSH="--load"
if [ -n "$PUSH" ]; then
    LOAD_OR_PUSH="--push"
fi

echo "[build] platform=$PLATFORM tag=$TAG prebuild_arena=$PREBUILD_ARENA action=$LOAD_OR_PUSH"

docker buildx build \
    --platform "$PLATFORM" \
    --build-arg "PREBUILD_ARENA=$PREBUILD_ARENA" \
    -t "$TAG" \
    -f docker/Dockerfile \
    $LOAD_OR_PUSH \
    "${EXTRA_ARGS[@]}" \
    .

echo
echo "[build] done."
if [ "$LOAD_OR_PUSH" = "--load" ]; then
    echo "[build] image loaded locally as $TAG"
    echo "[build] run on an x86 host:  docker run --gpus all --rm -it -p 8090:8090 $TAG"
else
    echo "[build] image pushed to $TAG"
    echo "[build] pull on the target host:  docker pull $TAG"
fi
