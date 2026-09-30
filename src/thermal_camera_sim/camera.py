"""Accès caméra (réelle ou synthétique) et export vidéo.

Cette couche est volontairement mince : elle ne fait que produire des frames
(webcam, image fixe ou source synthétique) et les faire passer dans
`converter.frame_to_thermal`. Toute la logique de rendu testable vit dans
`converter.py`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from thermal_camera_sim.converter import Colormap, frame_to_thermal

DEFAULT_DEMO_SIZE = (480, 640)  # (height, width), comme un frame OpenCV


def synthetic_heat_frame(
    step: int,
    size: tuple[int, int] = DEFAULT_DEMO_SIZE,
    n_blobs: int = 2,
) -> np.ndarray:
    """Génère une image grayscale synthétique avec des "sources de chaleur" mobiles.

    Ne nécessite aucune webcam — utilisé par le mode démo et par les tests.
    Chaque blob suit une trajectoire circulaire déterministe (fonction pure de
    `step`), donc le résultat est reproductible.
    """
    height, width = size
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    frame = np.zeros((height, width), dtype=np.float32)

    for blob_index in range(n_blobs):
        phase = step * 0.05 + blob_index * (2 * np.pi / max(n_blobs, 1))
        cx = width / 2 + (width / 3) * np.cos(phase)
        cy = height / 2 + (height / 3) * np.sin(phase * 1.3)
        radius = min(height, width) / 6
        # La gaussienne 2D est le produit de deux gaussiennes 1D : bien plus rapide.
        gy = np.exp(-((ys - cy) ** 2) / (2 * radius**2))
        gx = np.exp(-((xs - cx) ** 2) / (2 * radius**2))
        frame += 255.0 * np.outer(gy, gx)

    # Léger bruit de fond pour éviter un noir parfaitement uniforme.
    frame += 10.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def iter_synthetic_frames(
    n_frames: int, size: tuple[int, int] = DEFAULT_DEMO_SIZE
) -> Iterator[np.ndarray]:
    for step in range(n_frames):
        yield synthetic_heat_frame(step, size=size)


def load_image_frame(path: str | Path) -> np.ndarray:
    """Charge une image fixe depuis le disque (mode --source image)."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {path}")
    return frame


def export_thermal_video(
    output_path: str | Path,
    source: str = "demo",
    input_path: str | Path | None = None,
    colormap: Colormap = Colormap.IRON,
    contrast: str = "auto",
    blur_kernel: int = 9,
    n_frames: int = 90,
    fps: int = 30,
    device_index: int = 0,
) -> Path:
    """Exporte une vidéo thermique (mp4) depuis la webcam, une image fixe ou le mode démo.

    - source="demo" : `n_frames` générées synthétiquement (aucune caméra requise).
    - source="image" : boucle la même image `input_path` `n_frames` fois (utile
      pour prévisualiser une palette sur une capture existante).
    - source="webcam" : capture réelle pendant `n_frames` frames.
    """
    output_path = Path(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None

    def write_frame(raw_frame: np.ndarray) -> None:
        nonlocal writer
        thermal = frame_to_thermal(raw_frame, colormap, contrast, blur_kernel)
        if writer is None:
            height, width = thermal.shape[:2]
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        writer.write(thermal)

    if source == "demo":
        for frame in iter_synthetic_frames(n_frames):
            write_frame(frame)
    elif source == "image":
        if input_path is None:
            raise ValueError("--input est requis avec --source image")
        still = load_image_frame(input_path)
        for _ in range(n_frames):
            write_frame(still)
    elif source == "webcam":
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra #{device_index}.")
        try:
            for _ in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                write_frame(frame)
                time.sleep(0)
        finally:
            cap.release()
    else:
        raise ValueError(f"Source inconnue : {source!r}")

    if writer is not None:
        writer.release()
    return output_path
