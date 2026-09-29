"""Construit l'exécutable Windows autonome (dist/ThermalCameraSim.exe) avec PyInstaller.

Usage : python -m pip install -e ".[build]"  puis  python scripts/build_exe.py
L'icône est dessinée ici (palette Iron + réticule), aucune image externe n'est requise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def make_icon(path: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from thermal_camera_sim.converter import Colormap, palette_lut

    size = 256
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    heat = np.exp(-((xx - 150) ** 2 + (yy - 110) ** 2) / (2 * 55**2))
    heat += 0.7 * np.exp(-((xx - 90) ** 2 + (yy - 170) ** 2) / (2 * 40**2))
    gray = (np.clip(heat / heat.max(), 0, 1) * 255).astype(np.uint8)
    lut = palette_lut(Colormap.IRON).reshape(256, 3)[:, ::-1]  # BGR -> RGB
    rgb = lut[gray]

    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((8, 8, size - 8, size - 8), radius=52, fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    c, r = size // 2, 34
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line((c + dx * 14, c + dy * 14, c + dx * (r + 22), c + dy * (r + 22)), fill="white", width=9)
    d.ellipse((c - 8, c - 8, c + 8, c + 8), outline="white", width=5)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    icon = BUILD / "thermal.ico"
    make_icon(icon)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "ThermalCameraSim",
        "--icon",
        str(icon),
        "--paths",
        str(ROOT / "src"),
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(BUILD / "pyinstaller"),
        "--specpath",
        str(BUILD),
        str(ROOT / "src" / "thermal_camera_sim" / "__main__.py"),
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
