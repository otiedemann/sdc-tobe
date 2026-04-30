# Build pipeline for the Parrot Sphinx SDC arena scene (Path A).
#
# Targets:
#   pngs   — generate ArUco PNG bitmaps (needs cv2 + numpy in your venv)
#   fbxs   — wrap each PNG in a 1m FBX plane (needs Blender; calls headless)
#   yaml   — emit the Sphinx --config-file YAML
#   all    — pngs + fbxs + yaml
#   clean  — wipe out/
#
# Override BLENDER if Blender lives somewhere unusual:
#   make fbxs BLENDER=/Applications/Blender.app/Contents/MacOS/Blender

PY      ?= python3
BLENDER ?= blender

OUT     := out
PNG_DIR := $(OUT)/markers
FBX_DIR := $(OUT)/fbx
YAML    := $(OUT)/arena.yml

.PHONY: all pngs fbxs yaml clean check

all: yaml

pngs:
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)

fbxs: pngs
	mkdir -p $(FBX_DIR)
	$(BLENDER) --background --python build_marker_fbx.py -- \
	    --in-dir $(PNG_DIR) --out-dir $(FBX_DIR)

yaml: fbxs
	$(PY) arena_to_sphinx_yaml.py --fbx-dir $(FBX_DIR) --out $(YAML)

clean:
	rm -rf $(OUT)

# Self-test: generate the PNGs and YAML (skips Blender so this works
# anywhere the project's Python deps are installed). Use `make all` for
# the real pipeline.
check:
	$(PY) generate_aruco_pngs.py --out-dir $(PNG_DIR)
	$(PY) arena_to_sphinx_yaml.py --fbx-dir $(FBX_DIR) --out $(YAML)
	@echo
	@echo "self-test ok — PNGs in $(PNG_DIR), placeholder YAML in $(YAML)"
	@echo "(run 'make fbxs' on a host with Blender to produce the FBXs)"
