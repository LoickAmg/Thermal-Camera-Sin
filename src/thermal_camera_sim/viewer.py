"""Fenêtre interactive façon caméra thermique : webcam (ou scène de démo), palettes,
échelle de températures, réticule central, points chaud et froid, mesures au clic.

Tout reste une *simulation* : la « température » est tirée de la luminosité de l'image
(voir le README) — l'interface le rappelle en permanence.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from thermal_camera_sim.camera import synthetic_heat_frame
from thermal_camera_sim.converter import (
    SIMULATED_MAX_C,
    SIMULATED_MIN_C,
    Colormap,
    apply_blur,
    apply_colormap,
    apply_contrast,
    intensity_to_simulated_celsius,
    palette_lut,
    to_grayscale_intensity,
)

WINDOW = "Thermal Camera Sim"
VIEW_W, VIEW_H = 960, 720
TOP, BOTTOM, LEGEND = 52, 40, 96
BG = (18, 16, 14)  # BGR
TEXT = (235, 238, 240)
DIM = (150, 150, 145)
ACCENT = (40, 170, 255)  # orange (BGR)
HOT = (60, 60, 255)
COLD = (255, 170, 60)

CONTRAST_LABELS = {"auto": "Contraste auto", "equalize": "Égalisé", "none": "Brut"}
HELP_LINES = [
    ("P  ou  1–6", "changer de palette"),
    ("C", "mode de contraste"),
    ("+  /  −", "flou du capteur"),
    ("Clic gauche", "poser un point de mesure (3 max)"),
    ("Clic droit", "effacer les points"),
    ("W", "basculer webcam / démo"),
    ("S", "capture PNG (dossier Images)"),
    ("F", "plein écran"),
    ("H", "afficher / masquer l'aide"),
    ("Q  ou  Échap", "quitter"),
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            path = folder / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def open_webcam(index: int = 0) -> cv2.VideoCapture | None:
    """Ouvre la webcam (DirectShow sous Windows : bien plus rapide et fiable que MSMF)."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def celsius(value: float) -> str:
    return f"{value:.1f} °C".replace(".", ",")


@dataclass
class ViewerState:
    palette: Colormap = Colormap.IRON
    contrast: str = "auto"
    blur: int = 9
    show_help: bool = True
    fullscreen: bool = False
    pins: list[tuple[int, int]] = field(default_factory=list)
    toast: str = ""
    toast_until: float = 0.0


def cycle(values: list, current):
    return values[(values.index(current) + 1) % len(values)]


class ThermalViewer:
    def __init__(
        self, device: int = 0, prefer_webcam: bool = True, still: np.ndarray | None = None
    ) -> None:
        self.device = device
        self.still = still
        self.state = ViewerState()
        self.cap = open_webcam(device) if prefer_webcam else None
        self.step = 0
        self.fps = 0.0
        self.fonts = {
            "title": _font(19, bold=True),
            "ui": _font(15),
            "ui_bold": _font(15, bold=True),
            "small": _font(12),
            "tag": _font(13, bold=True),
        }
        self.frame_shape = (480, 640)
        if prefer_webcam and self.cap is None and still is None:
            self.notify("Aucune webcam détectée : scène de démonstration")

    # ------------------------------------------------------------------ sources

    @property
    def source_label(self) -> str:
        if self.still is not None:
            return "IMAGE"
        return "WEBCAM" if self.cap is not None else "DÉMO"

    def next_frame(self) -> np.ndarray:
        if self.still is not None:
            return self.still
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                return frame
            self.cap.release()
            self.cap = None
            self.notify("Webcam perdue : scène de démonstration")
        self.step += 1
        return synthetic_heat_frame(self.step, n_blobs=3)

    def toggle_source(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.notify("Scène de démonstration")
            return
        self.cap = open_webcam(self.device)
        self.notify("Webcam" if self.cap is not None else "Aucune webcam disponible")

    def notify(self, message: str, seconds: float = 2.5) -> None:
        self.state.toast = message
        self.state.toast_until = time.monotonic() + seconds

    # ------------------------------------------------------------------ rendu

    def render(self, raw: np.ndarray) -> np.ndarray:
        st = self.state
        gray_raw = apply_blur(to_grayscale_intensity(raw), st.blur)
        self.frame_shape = gray_raw.shape
        shown = apply_contrast(gray_raw, st.contrast)
        colored = apply_colormap(shown, st.palette)
        view = cv2.resize(colored, (VIEW_W, VIEW_H), interpolation=cv2.INTER_CUBIC)

        h, w = gray_raw.shape
        sx, sy = VIEW_W / w, VIEW_H / h
        lo, hi = int(gray_raw.min()), int(gray_raw.max())
        cold_idx = np.unravel_index(int(np.argmin(gray_raw)), gray_raw.shape)
        hot_idx = np.unravel_index(int(np.argmax(gray_raw)), gray_raw.shape)

        canvas = np.full((TOP + VIEW_H + BOTTOM, VIEW_W + LEGEND, 3), BG, dtype=np.uint8)
        canvas[TOP : TOP + VIEW_H, :VIEW_W] = view
        self._draw_legend(canvas, lo, hi)

        labels: list[tuple[tuple[int, int], str, tuple[int, int, int], str]] = []

        def to_view(y: int, x: int) -> tuple[int, int]:
            return int(x * sx), int(y * sy) + TOP

        # Point chaud et point froid
        for (y, x), color, tag in ((hot_idx, HOT, "MAX"), (cold_idx, COLD, "MIN")):
            px, py = to_view(int(y), int(x))
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 8, color, 2, cv2.LINE_AA)
            labels.append(
                (
                    (px + 14, py - 10),
                    f"{tag} {celsius(intensity_to_simulated_celsius(int(gray_raw[y, x])))}",
                    color,
                    "tag",
                )
            )

        # Réticule central
        cx, cy = VIEW_W // 2, TOP + VIEW_H // 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(
                canvas,
                (cx + dx * 6, cy + dy * 6),
                (cx + dx * 22, cy + dy * 22),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        center_c = intensity_to_simulated_celsius(int(gray_raw[h // 2, w // 2]))
        labels.append(((cx + 26, cy + 8), celsius(center_c), (255, 255, 255), "ui_bold"))

        # Points de mesure posés au clic
        for i, (x, y) in enumerate(st.pins, start=1):
            if 0 <= x < w and 0 <= y < h:
                px, py = to_view(y, x)
                cv2.drawMarker(
                    canvas, (px, py), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA
                )
                temp = intensity_to_simulated_celsius(int(gray_raw[y, x]))
                labels.append(((px + 12, py + 4), f"P{i}  {celsius(temp)}", (255, 255, 255), "tag"))

        return self._draw_text(canvas, labels, lo, hi)

    def _draw_legend(self, canvas: np.ndarray, lo: int, hi: int) -> None:
        x0 = VIEW_W + 34
        top, bottom = TOP + 40, TOP + VIEW_H - 40
        lut = palette_lut(self.state.palette).reshape(256, 3)
        for y in range(top, bottom):
            t = 1 - (y - top) / (bottom - top)
            canvas[y, x0 : x0 + 22] = lut[int(t * 255)]
        cv2.rectangle(canvas, (x0 - 1, top - 1), (x0 + 22, bottom), (90, 90, 90), 1)

    def _draw_text(self, canvas, labels, lo: int, hi: int) -> np.ndarray:
        st = self.state
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        f = self.fonts
        rgb = lambda c: (c[2], c[1], c[0])  # noqa: E731

        # Barre du haut
        d.text((16, 14), "THERMAL CAM", font=f["title"], fill=rgb(TEXT))
        title_end = d.textbbox((16, 14), "THERMAL CAM ", font=f["title"])[2]
        d.text((title_end, 14), "SIM", font=f["title"], fill=rgb(ACCENT))
        x = 210
        for text, color in (
            (self.source_label, ACCENT),
            (st.palette.label, TEXT),
            (CONTRAST_LABELS[st.contrast], TEXT),
            (f"Flou {st.blur}", TEXT),
            (f"{self.fps:.0f} ips", DIM),
        ):
            box = d.textbbox((x, 17), text, font=f["tag"])
            d.rounded_rectangle(
                (box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5),
                radius=4,
                outline=rgb((70, 70, 70)),
            )
            d.text((x, 17), text, font=f["tag"], fill=rgb(color))
            x = box[2] + 20

        # Échelle : températures extrêmes de l'image
        lo_c = intensity_to_simulated_celsius(lo) if st.contrast != "none" else SIMULATED_MIN_C
        hi_c = intensity_to_simulated_celsius(hi) if st.contrast != "none" else SIMULATED_MAX_C
        d.text((VIEW_W + 14, TOP + 12), celsius(hi_c), font=f["ui_bold"], fill=rgb(TEXT))
        d.text((VIEW_W + 14, TOP + VIEW_H - 32), celsius(lo_c), font=f["ui_bold"], fill=rgb(TEXT))

        # Étiquettes des marqueurs (avec ombre pour rester lisibles sur toutes les palettes)
        for (lx, ly), text, color, font in labels:
            d.text((lx + 1, ly + 1), text, font=f[font], fill=(0, 0, 0))
            d.text((lx, ly), text, font=f[font], fill=rgb(color))

        # Barre du bas : rappel honnête
        d.text(
            (16, TOP + VIEW_H + 11),
            "Simulation : les couleurs viennent de la luminosité de l'image, "
            "pas d'une vraie mesure infrarouge.",
            font=f["small"],
            fill=rgb(DIM),
        )
        d.text(
            (VIEW_W - 150, TOP + VIEW_H + 11),
            "H : aide  ·  Q : quitter",
            font=f["small"],
            fill=rgb(DIM),
        )

        if st.show_help:
            pad, line_h = 18, 24
            w_box, h_box = 360, pad * 2 + line_h * (len(HELP_LINES) + 1)
            x0, y0 = 20, TOP + 20
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rounded_rectangle(
                (x0, y0, x0 + w_box, y0 + h_box), radius=10, fill=(10, 12, 16, 215)
            )
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            d = ImageDraw.Draw(img)
            d.text((x0 + pad, y0 + pad - 2), "Commandes", font=f["ui_bold"], fill=rgb(ACCENT))
            for i, (key, what) in enumerate(HELP_LINES, start=1):
                y = y0 + pad + i * line_h
                d.text((x0 + pad, y), key, font=f["ui_bold"], fill=rgb(TEXT))
                d.text((x0 + pad + 120, y), what, font=f["ui"], fill=rgb((200, 200, 200)))

        if st.toast and time.monotonic() < st.toast_until:
            box = d.textbbox((0, 0), st.toast, font=f["ui_bold"])
            tw = box[2] - box[0]
            x0 = (VIEW_W - tw) // 2
            y0 = TOP + VIEW_H - 64
            d.rounded_rectangle(
                (x0 - 16, y0 - 10, x0 + tw + 16, y0 + 30), radius=8, fill=(20, 22, 28)
            )
            d.text((x0, y0), st.toast, font=f["ui_bold"], fill=rgb(TEXT))

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ interactions

    def on_mouse(self, event: int, x: int, y: int, *_args) -> None:
        h, w = self.frame_shape
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < VIEW_W and TOP <= y < TOP + VIEW_H:
            fx, fy = int(x * w / VIEW_W), int((y - TOP) * h / VIEW_H)
            self.state.pins = (self.state.pins + [(fx, fy)])[-3:]
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.state.pins = []

    def handle_key(self, key: int) -> bool:
        """Applique une touche ; renvoie False pour quitter."""
        st = self.state
        palettes = list(Colormap)
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("p"), ord("P")):
            st.palette = cycle(palettes, st.palette)
            self.notify(f"Palette : {st.palette.label}", 1.2)
        elif ord("1") <= key <= ord(str(len(palettes))):
            st.palette = palettes[key - ord("1")]
            self.notify(f"Palette : {st.palette.label}", 1.2)
        elif key in (ord("c"), ord("C")):
            st.contrast = cycle(["auto", "equalize", "none"], st.contrast)
            self.notify(CONTRAST_LABELS[st.contrast], 1.2)
        elif key in (ord("+"), ord("=")):
            st.blur = min(31, st.blur + 2)
        elif key in (ord("-"), ord("_")):
            st.blur = max(0, st.blur - 2)
        elif key in (ord("h"), ord("H")):
            st.show_help = not st.show_help
        elif key in (ord("w"), ord("W")):
            self.toggle_source()
        elif key in (ord("f"), ord("F")):
            st.fullscreen = not st.fullscreen
            cv2.setWindowProperty(
                WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if st.fullscreen else cv2.WINDOW_NORMAL,
            )
        return True

    def save_snapshot(self, image: np.ndarray) -> Path:
        folder = Path.home() / "Pictures" / "Thermal Camera Sim"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"thermique-{datetime.now():%Y%m%d-%H%M%S}.png"
        cv2.imencode(".png", image)[1].tofile(str(path))
        return path

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, VIEW_W + LEGEND, TOP + VIEW_H + BOTTOM)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        last = time.perf_counter()
        try:
            while True:
                image = self.render(self.next_frame())
                cv2.imshow(WINDOW, image)
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - last, 1e-3))
                last = now
                if self.cap is None:
                    time.sleep(1 / 45)
                key = cv2.waitKeyEx(1)
                if key != -1:
                    key &= 0xFF
                    if key in (ord("s"), ord("S")):
                        self.notify(f"Capture enregistrée : {self.save_snapshot(image).name}")
                    elif not self.handle_key(key):
                        break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
