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
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    frame = np.zeros((height, width), dtype=np.float32)

    for blob_index in range(n_blobs):
        phase = step * 0.05 + blob_index * (2 * np.pi / max(n_blobs, 1))
        cx = width / 2 + (width / 3) * np.cos(phase)
        cy = height / 2 + (height / 3) * np.sin(phase * 1.3)
        radius = min(height, width) / 6
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        frame += 255.0 * np.exp(-dist_sq / (2 * radius**2))

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


def stream_webcam(
    colormap: Colormap = Colormap.IRON,
    contrast: str = "auto",
    blur_kernel: int = 9,
    device_index: int = 0,
    window_name: str = "thermal-camera-sim",
) -> None:
    """Affiche en direct le rendu thermique de la webcam. Nécessite une vraie caméra.

    Lève RuntimeError si la caméra ne peut pas être ouverte, plutôt que de
    planter silencieusement ou de boucler sur des frames vides — même
    contrat que ascii-camera.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra #{device_index}. "
            "Utilisez --source demo pour tester sans webcam."
        )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Échec de lecture d'une frame webcam.")
            thermal = frame_to_thermal(frame, colormap, contrast, blur_kernel)
            cv2.imshow(window_name, thermal)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # 'q' ou Échap
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


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
