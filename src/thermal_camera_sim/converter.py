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
    """Palettes disponibles, mappées vers les constantes OpenCV correspondantes."""

    IRON = "iron"  # palette "fer" façon FLIR (noir -> violet -> orange -> blanc)
    RAINBOW = "rainbow"
    GRAYSCALE = "grayscale"

    def to_cv2(self) -> int | None:
        return {
            Colormap.IRON: cv2.COLORMAP_INFERNO,
            Colormap.RAINBOW: cv2.COLORMAP_RAINBOW,
            Colormap.GRAYSCALE: None,
        }[self]


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
    cv2_map = colormap.to_cv2()
    if cv2_map is None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, cv2_map)


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
