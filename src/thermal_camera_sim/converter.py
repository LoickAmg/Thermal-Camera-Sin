"""Conversion d'une image ordinaire en rendu "thermique" simulé.

Toute la logique ici est pure (numpy/OpenCV, aucun accès caméra) afin d'être
testable sans matériel et sans fenêtre graphique.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# Plage de "température" simulée (°C) associée linéairement à l'intensité
# 0-255 d'une image en niveaux de gris. Purement arbitraire — voir le README.
SIMULATED_MIN_C = 15.0
SIMULATED_MAX_C = 45.0


class Colormap(str, Enum):
    """Palettes des caméras thermiques du commerce."""

    IRON = "iron"  # « fer » façon FLIR : noir -> bleu nuit -> magenta -> orange -> jaune -> blanc
    RAINBOW = "rainbow"  # arc-en-ciel haute sensibilité
    GRAYSCALE = "grayscale"  # « blanc = chaud »
    BLACK_HOT = "black-hot"  # « noir = chaud », prisé en vision nocturne
    ARCTIC = "arctic"  # bleus froids, chaud en ambre
    LAVA = "lava"  # noir -> rouge -> jaune

    @property
    def label(self) -> str:
        return {
            Colormap.IRON: "Iron",
            Colormap.RAINBOW: "Rainbow",
            Colormap.GRAYSCALE: "White hot",
            Colormap.BLACK_HOT: "Black hot",
            Colormap.ARCTIC: "Arctic",
            Colormap.LAVA: "Lava",
        }[self]


# Points de couleur (position 0-1, couleur RGB) des palettes dessinées à la main.
_PALETTE_STOPS: dict[Colormap, list[tuple[float, tuple[int, int, int]]]] = {
    Colormap.IRON: [
        (0.0, (0, 0, 0)),
        (0.15, (20, 10, 90)),
        (0.35, (140, 20, 150)),
        (0.55, (225, 60, 50)),
        (0.72, (250, 140, 10)),
        (0.88, (255, 220, 60)),
        (1.0, (255, 255, 240)),
    ],
    Colormap.ARCTIC: [
        (0.0, (5, 10, 40)),
        (0.3, (20, 70, 170)),
        (0.6, (90, 190, 240)),
        (0.8, (230, 245, 255)),
        (0.92, (255, 200, 80)),
        (1.0, (255, 140, 20)),
    ],
    Colormap.LAVA: [
        (0.0, (0, 0, 0)),
        (0.3, (110, 0, 10)),
        (0.6, (230, 40, 10)),
        (0.85, (255, 190, 30)),
        (1.0, (255, 255, 200)),
    ],
}


def palette_lut(colormap: Colormap) -> np.ndarray:
    """Table de correspondance 256 × 1 × 3 (BGR) d'une palette."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    if colormap == Colormap.GRAYSCALE:
        return cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR).reshape(256, 1, 3)
    if colormap == Colormap.BLACK_HOT:
        return cv2.cvtColor(255 - ramp, cv2.COLOR_GRAY2BGR).reshape(256, 1, 3)
    if colormap == Colormap.RAINBOW:
        return cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(256, 1, 3)
    stops = _PALETTE_STOPS[colormap]
    xs = np.array([s[0] for s in stops]) * 255
    positions = np.arange(256)
    rgb = np.stack([np.interp(positions, xs, [s[1][ch] for s in stops]) for ch in range(3)], axis=1)
    return rgb[:, ::-1].astype(np.uint8).reshape(256, 1, 3)


_LUT_CACHE: dict[Colormap, np.ndarray] = {}


def to_grayscale_intensity(frame: np.ndarray) -> np.ndarray:
    """Réduit une image (couleur ou déjà grayscale) à une carte d'intensité 2D uint8."""
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Forme d'image inattendue : {frame.shape}")
    return gray.astype(np.uint8)


def stretch_contrast(gray: np.ndarray) -> np.ndarray:
    """Étire linéairement l'intensité sur [0, 255] selon le min/max réel de l'image.

    Robuste à une image totalement plate (min == max) : retourne l'image
    inchangée plutôt que de diviser par zéro.
    """
    lo, hi = float(gray.min()), float(gray.max())
    if hi <= lo:
        return gray.copy()
    stretched = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Égalisation d'histogramme (contraste local, plus agressif que l'étirement)."""
    return cv2.equalizeHist(gray)


def apply_contrast(gray: np.ndarray, mode: str = "auto") -> np.ndarray:
    if mode == "auto":
        return stretch_contrast(gray)
    if mode == "equalize":
        return equalize_histogram(gray)
    if mode == "none":
        return gray
    raise ValueError(f"Mode de contraste inconnu : {mode!r}")


def apply_blur(gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Flou gaussien simulant la résolution/le bruit plus faibles d'un capteur thermique.

    kernel_size <= 1 désactive le flou (pas d'appel OpenCV, évite l'erreur
    "kernel size must be positive and odd" sur une valeur de 0).
    """
    if kernel_size <= 1:
        return gray
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def apply_colormap(gray: np.ndarray, colormap: Colormap) -> np.ndarray:
    """Applique une palette "thermique" à une image en niveaux de gris.

    Retourne toujours une image BGR 3 canaux (même en mode GRAYSCALE, pour que
    l'appelant puisse dessiner du texte/overlay couleur dessus uniformément).
    """
    lut = _LUT_CACHE.get(colormap)
    if lut is None:
        lut = _LUT_CACHE[colormap] = palette_lut(colormap)
    return cv2.applyColorMap(gray, lut)


def intensity_to_simulated_celsius(value: int) -> float:
    """Correspondance linéaire arbitraire intensité [0,255] -> température simulée."""
    ratio = value / 255.0
    return SIMULATED_MIN_C + ratio * (SIMULATED_MAX_C - SIMULATED_MIN_C)


@dataclass(frozen=True)
class Hotspot:
    x: int
    y: int
    intensity: int
    simulated_celsius: float


def detect_hotspot(gray: np.ndarray) -> Hotspot:
    """Trouve le pixel le plus "chaud" (intensité max) de l'image."""
    if gray.size == 0:
        raise ValueError("Image vide")
    idx = int(np.argmax(gray))
    y, x = np.unravel_index(idx, gray.shape)
    intensity = int(gray[y, x])
    return Hotspot(
        x=int(x),
        y=int(y),
        intensity=intensity,
        simulated_celsius=intensity_to_simulated_celsius(intensity),
    )


def draw_hotspot_marker(frame_bgr: np.ndarray, hotspot: Hotspot) -> np.ndarray:
    """Dessine un réticule + étiquette de température (simulée) sur le point chaud."""
    out = frame_bgr.copy()
    x, y = hotspot.x, hotspot.y
    size = 10
    color = (255, 255, 255)
    cv2.line(out, (x - size, y), (x + size, y), color, 1)
    cv2.line(out, (x, y - size), (x, y + size), color, 1)
    cv2.circle(out, (x, y), size + 4, color, 1)
    label = f"{hotspot.simulated_celsius:.1f}C (simule)"
    text_y = y - size - 6 if y - size - 6 > 10 else y + size + 16
    cv2.putText(
        out,
        label,
        (max(x - 40, 2), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def frame_to_thermal(
    frame: np.ndarray,
    colormap: Colormap = Colormap.IRON,
    contrast: str = "auto",
    blur_kernel: int = 9,
    mark_hotspot: bool = True,
) -> np.ndarray:
    """Pipeline complet : image source -> rendu thermique simulé BGR.

    C'est la fonction utilisée à la fois par le flux webcam live et par
    l'export vidéo — un seul chemin de code pour les deux, testé ici en
    isolation sans caméra.
    """
    gray = to_grayscale_intensity(frame)
    gray = apply_blur(gray, blur_kernel)
    gray = apply_contrast(gray, contrast)
    colored = apply_colormap(gray, colormap)
    if mark_hotspot:
        hotspot = detect_hotspot(gray)
        colored = draw_hotspot_marker(colored, hotspot)
    return colored
