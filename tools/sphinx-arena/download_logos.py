#!/usr/bin/env python3
"""Download arena wall-painting logos and convert to PNG.

Logos:
  - MBDA Logo (SVG from Wikipedia commons)
  - brigk Logo (SVG from brigk.digital)
  - Swarm Drone Challenge Logo (PNG)
  - Team logo (placeholder; drop your real PNG at out/logos/team.png to override)

SVGs are rasterised via ``cairosvg`` (pip-installable) with fallback to
``rsvg-convert`` (libsrvg2-bin) or ImageMagick ``convert``.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# (name, source URL, source format)
# MBDA's URL on Wikipedia goes through a redirect; the actual file lives
# on commons. Fall back to direct commons URL if the wikipedia /Datei:
# link fails.
LOGOS = [
    {
        "name": "mbda",
        # Real Wikipedia commons path (verified by parsing the
        # de.wikipedia.org/Datei: page — the path is a hash of the
        # filename, can't be guessed).
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/35/MBDA-Logo.svg",
        "format": "svg",
    },
    {
        "name": "brigk",
        "url": "https://www.brigk.digital/wp-content/uploads/2025/04/brigk_black.svg",
        "format": "svg",
    },
    {
        "name": "sdc",
        "url": "https://swarmdronechallenge.digital/assets/sdc-logo-D8_MXqOZ.png",
        "format": "png",
    },
]

USER_AGENT = "Mozilla/5.0 (sphinx-arena logo fetcher; SDC26)"


def fetch(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        dst.write_bytes(r.read())


def svg_to_png(svg_path: Path, png_path: Path, target_width: int = 2048) -> None:
    """Convert SVG → PNG. Try cairosvg first; fall back to rsvg-convert
    or ImageMagick convert."""
    # cairosvg
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=target_width,
        )
        return
    except ImportError:
        pass

    # rsvg-convert (libsrvg2-bin on Ubuntu)
    if shutil.which("rsvg-convert"):
        subprocess.check_call([
            "rsvg-convert", "-w", str(target_width),
            "-o", str(png_path), str(svg_path),
        ])
        return

    # ImageMagick
    if shutil.which("convert"):
        subprocess.check_call([
            "convert", "-density", "300",
            "-resize", f"{target_width}x",
            str(svg_path), str(png_path),
        ])
        return

    raise RuntimeError(
        "no SVG → PNG converter found. Install one of:\n"
        "  pip install cairosvg     (recommended)\n"
        "  apt install librsvg2-bin\n"
        "  apt install imagemagick"
    )


def write_placeholder_png(path: Path, side_px: int = 1024) -> None:
    """Hand-craft a uniform-light-grey PNG via stdlib zlib + struct.

    Used when the user hasn't supplied a team logo yet — the painting
    framework still works; this is just a visible "drop your logo
    here" placeholder.
    """
    import struct
    import zlib
    color = (210, 220, 235)  # light blue-grey
    ihdr = struct.pack(">IIBBBBB", side_px, side_px, 8, 2, 0, 0, 0)
    row = bytes([0]) + bytes(color) * side_px
    raw = row * side_px
    idat = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b""))
    path.write_bytes(blob)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for downloaded logos.")
    p.add_argument("--target-width", type=int, default=2048,
                   help="PNG width to rasterise SVGs at.")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if cached PNG already exists.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for logo in LOGOS:
        name = logo["name"]
        png_path = args.out_dir / f"{name}.png"
        if png_path.exists() and not args.force:
            print(f"  cached: {png_path.name}")
            continue
        try:
            if logo["format"] == "svg":
                src = args.out_dir / f"{name}.svg"
                print(f"  fetch  {logo['url']}")
                fetch(logo["url"], src)
                svg_to_png(src, png_path, target_width=args.target_width)
            else:  # png
                print(f"  fetch  {logo['url']}")
                fetch(logo["url"], png_path)
            print(f"  ✓ {png_path.name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}", file=sys.stderr)

    # Team-logo placeholder. The operator can drop their real PNG at
    # out/logos/team.png to override; the build pipeline will pick up
    # whatever's there.
    team_path = args.out_dir / "team.png"
    if not team_path.exists():
        write_placeholder_png(team_path)
        print(f"  ✓ {team_path.name} (placeholder — drop your team's "
              f"PNG here to override)")
    else:
        print(f"  cached: {team_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
