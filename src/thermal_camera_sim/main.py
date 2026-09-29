"""CLI de thermal-camera-sim."""

from __future__ import annotations

import argparse
import sys

from thermal_camera_sim.camera import export_thermal_video, load_image_frame
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
        choices=["auto", "webcam", "demo", "image"],
        default="auto",
        help=(
            "Source des frames : auto (webcam si disponible, sinon démo — défaut), webcam, "
            "démo synthétique ou image fixe."
        ),
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
            source="demo" if args.source == "auto" else args.source,
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

    # Sans --output : la fenêtre interactive (c'est aussi ce que lance l'exécutable).
    from thermal_camera_sim.viewer import ThermalViewer

    still = None
    if args.source == "image":
        if not args.input:
            print("--input est requis avec --source image", file=sys.stderr)
            return 1
        try:
            still = load_image_frame(args.input)
        except FileNotFoundError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            return 1
    viewer = ThermalViewer(
        device=args.device, prefer_webcam=args.source in ("auto", "webcam"), still=still
    )
    viewer.state.palette = colormap
    viewer.state.contrast = args.contrast
    viewer.state.blur = max(0, args.blur)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
