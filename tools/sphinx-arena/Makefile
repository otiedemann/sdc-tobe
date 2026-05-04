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
FBX_DIR := $(OUT)/fbx
YAML    := $(OUT)/arena.yml
REPO_ROOT := $(realpath $(CURDIR)/../..)

.PHONY: all pngs fbxs arena-static yaml clean check

all: yaml

pngs:
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)

fbxs: pngs
	mkdir -p $(FBX_DIR)
	$(BLENDER) --background --python build_marker_fbx.py -- \
	    --in-dir $(PNG_DIR) --out-dir $(FBX_DIR)

# Combined static arena — floor + walls + pillars all baked into a
# single FBX with positions and dimensions in Blender vertex space.
# YAML places it at world origin with no scaling/rotation; eliminates
# the per-mesh transform quirks that were tilting walls and dropping
# materials.
arena-static: $(YAML_DEP)
	mkdir -p $(FBX_DIR)
	$(BLENDER) --background --python build_full_arena.py -- \
	    --arena-config $(REPO_ROOT)/controller_unified/arena_config.json \
	    --out $(FBX_DIR)/arena_static.fbx

yaml: fbxs arena-static
	$(PY) arena_to_sphinx_yaml.py --fbx-dir $(FBX_DIR) --out $(YAML) \
	    --emit-arena-static

clean:
	rm -rf $(OUT)

# Self-test: generate PNGs and emit the YAML (skips Blender so this
# works anywhere the project's Python deps are installed). For the
# real arena build run 'make all' on a host with Blender installed.
check:
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)
	$(PY) arena_to_sphinx_yaml.py --fbx-dir $(FBX_DIR) --out $(YAML) \
	    --emit-arena-static
	@echo
	@echo "self-test ok — PNGs in $(PNG_DIR), arena YAML in $(YAML)"
	@echo "(run 'make all' on a host with Blender to produce the FBXs)"
