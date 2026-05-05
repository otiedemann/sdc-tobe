# Build pipeline for the Parrot Sphinx SDC arena scene (Path A).
#
# Targets:
#   pngs      — generate ArUco PNG bitmaps (needs cv2 + numpy in your venv)
#   fbxs      — wrap each PNG in a 1m FBX plane (needs Blender; headless)
#   geometry  — generate floor.fbx + pillar.fbx primitives (needs Blender)
#   yaml      — emit the Sphinx --config-file YAML (markers + floor + pillars)
#   all       — pngs + fbxs + geometry + yaml
#   clean     — wipe out/
#
# Override BLENDER if Blender lives somewhere unusual:
#   make all BLENDER=/Applications/Blender.app/Contents/MacOS/Blender

PY      ?= python3
BLENDER ?= blender

OUT     := out
PNG_DIR := $(OUT)/markers
LOGO_DIR := $(OUT)/logos
FBX_DIR := $(OUT)/fbx
YAML    := $(OUT)/arena.yml
REPO_ROOT := $(realpath $(CURDIR)/../..)

.PHONY: all pngs fbxs arena-static logos paintings yaml rebuild clean check view-arena edit-arena

# `make all` rebuilds everything from sources (matches the historical
# behaviour). For just regenerating the YAML after editing the emitter,
# use `make yaml` directly.
all: rebuild

pngs:
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)

# fbxs depends on the marker PNG directory existing — not the (.PHONY)
# `pngs` target. That way re-running `make fbxs` after pngs are already
# built doesn't try to re-import cv2 (which isn't installed on every
# host that has Blender). Run `make pngs` explicitly when ArUco PNGs
# need to change.
fbxs: $(PNG_DIR)
	mkdir -p $(FBX_DIR)
	$(BLENDER) --background --python build_marker_fbx.py -- \
	    --in-dir $(PNG_DIR) --out-dir $(FBX_DIR)

# Auto-create the PNGs only on the very first run (when the directory
# doesn't exist yet). After that re-running `make fbxs` skips this and
# uses whatever's in $(PNG_DIR).
$(PNG_DIR):
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)

# Combined static arena — floor + walls + pillars all baked into a
# single FBX with positions and dimensions in Blender vertex space.
arena-static:
	mkdir -p $(FBX_DIR)
	$(BLENDER) --background --python build_full_arena.py -- \
	    --arena-config $(REPO_ROOT)/controller_unified/arena_config.json \
	    --out $(FBX_DIR)/arena_static.fbx

# Wall-painting logos: download external PNGs/SVGs into out/logos/,
# then build one FBX per logo for Sphinx mesh injection.
logos:
	$(PY) download_logos.py --out-dir $(LOGO_DIR)

# Same pattern as `fbxs`: depend on the directory rather than the .PHONY
# logos target so a re-run doesn't try to re-fetch over the network.
paintings: $(LOGO_DIR)
	mkdir -p $(FBX_DIR)
	$(BLENDER) --background --python build_painting_fbxs.py -- \
	    --in-dir $(LOGO_DIR) --out-dir $(FBX_DIR)

$(LOGO_DIR):
	$(PY) download_logos.py --out-dir $(LOGO_DIR)

# `yaml` is just the YAML emit — it does not force-rebuild the FBXs.
# Use `make all` (or the individual targets) when you want to rebuild
# meshes; `make yaml` is the fast path when you've only changed the
# YAML emitter logic.
#
# Default emits: floor + pillars + walls (each a single-mesh,
# single-material FBX scaled per-axis in YAML — Sphinx renders these
# reliably). The combined arena_static.fbx path is available via
# --emit-arena-static but Sphinx mesh-injection doesn't render its
# multi-mesh + multi-material content, so we don't use it by default.
# Paintings (MBDA / brigk / SDC / team logos) are intentionally NOT
# emitted by default — drop --emit-paintings here, or pass it on the
# command line, to add them back. The painting FBXs themselves are
# still built by `make paintings` so re-enabling is one flag away.
yaml:
	$(PY) arena_to_sphinx_yaml.py --fbx-dir $(FBX_DIR) --out $(YAML) \
	    --emit-floor --emit-pillars --emit-walls --emit-paintings \
	    --emit-actor-paths

# Convenience target: rebuild EVERYTHING from sources. Run this on a
# host that has cv2, Blender, and network access (for logo fetch).
rebuild: pngs fbxs arena-static logos paintings yaml

# Open the arena (existing or freshly-built) in Blender's GUI for
# inspection or editing. Two modes:
#   make view-arena   — open the .blend file directly (read-only-ish;
#                       just inspect existing geometry).
#   make edit-arena   — re-run the build script under Blender GUI so
#                       you get the freshly-generated scene (lets you
#                       tweak the python and immediately see results).
view-arena:
	@if [ ! -f $(FBX_DIR)/arena_static.blend ]; then \
	  echo "no $(FBX_DIR)/arena_static.blend yet — run 'make arena-static' first."; \
	  exit 1; \
	fi
	$(BLENDER) $(FBX_DIR)/arena_static.blend

edit-arena:
	mkdir -p $(FBX_DIR)
	$(BLENDER) --python build_full_arena.py -- \
	    --arena-config $(REPO_ROOT)/controller_unified/arena_config.json \
	    --out $(FBX_DIR)/arena_static.fbx \
	    --gui --no-fbx

clean:
	rm -rf $(OUT)

# Self-test: generate PNGs and emit the YAML (skips Blender so this
# works anywhere the project's Python deps are installed). For the
# real arena build run 'make all' on a host with Blender installed.
check:
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)
	$(PY) arena_to_sphinx_yaml.py --fbx-dir $(FBX_DIR) --out $(YAML) \
	    --emit-arena-static --emit-paintings
	@echo
	@echo "self-test ok — PNGs in $(PNG_DIR), arena YAML in $(YAML)"
	@echo "(run 'make all' on a host with Blender to produce the FBXs)"
