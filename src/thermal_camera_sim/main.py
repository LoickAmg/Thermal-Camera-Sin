"""CLI de thermal-camera-sim."""

from __future__ import annotations

import argparse
import sys

from thermal_camera_sim.camera import export_thermal_video, stream_webcam
from thermal_camera_sim.converter import Colormap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-camera-sim",
        description=(
            "Simulation de caméra thermique (colormaps, flou, point chaud). "
            "Ce n'est PAS un vrai capteur thermique — voir le README."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "demo", "image"],
        default="demo",
        help="Source des frames : webcam réelle, démo synthétique (défaut), ou image fixe.",
    )
    parser.add_argument("--input", help="Chemin de l'image (requis avec --source image).")
    parser.add_argument(
        "--colormap",
        choices=[c.value for c in Colormap],
        default=Colormap.IRON.value,
        help="Palette thermique (défaut: iron).",
    )
    parser.add_argument(
        "--contrast",
        choices=["auto", "equalize", "none"],
        default="auto",
        help="Mode de contraste (défaut: auto = étirement min/max).",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=9,
        help="Taille du noyau de flou gaussien, 0 pour désactiver (défaut: 9).",
    )
    parser.add_argument(
        "--output",
        help="Si fourni, exporte une vidéo mp4 au lieu d'afficher une fenêtre live.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=90,
        help="Nombre de frames à exporter avec --output (défaut: 90).",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS de la vidéo exportée.")
    parser.add_argument("--device", type=int, default=0, help="Index de la webcam (défaut: 0).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    colormap = Colormap(args.colormap)

    if args.output:
        path = export_thermal_video(
            output_path=args.output,
            source=args.source,
            input_path=args.input,
            colormap=colormap,
            contrast=args.contrast,
            blur_kernel=args.blur,
            n_frames=args.frames,
            fps=args.fps,
            device_index=args.device,
        )
        print(f"Vidéo thermique exportée : {path}")
        return 0

    if args.source != "webcam":
        print(
            "L'affichage live n'est disponible qu'avec --source webcam. "
            "Utilisez --output pour exporter une vidéo en mode demo/image.",
            file=sys.stderr,
        )
        return 1

    try:
        stream_webcam(
            colormap=colormap,
            contrast=args.contrast,
            blur_kernel=args.blur,
            device_index=args.device,
        )
    except RuntimeError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
