"""Fenêtre interactive façon caméra thermique : webcam (ou scène de démo), palettes,
échelle de températures, réticule central, points chaud et froid, mesures au clic.

Tout reste une *simulation* : la « température » est tirée de la luminosité de l'image
(voir le README) — l'interface le rappelle en permanence.

L'image est recomposée à la taille réelle de la fenêtre à chaque trame : en plein écran,
le texte et les marqueurs sont dessinés à la résolution de l'écran au lieu d'être étirés.
Le processus se déclare « DPI-aware » sous Windows : sinon le système agrandit toute la
fenêtre (125 %, 150 %…) en l'étirant comme une image, ce qui la rend floue.
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
# Taille de référence de la fenêtre : toutes les longueurs ci-dessous y sont exprimées,
# puis mises à l'échelle de la fenêtre réelle (voir Layout).
BASE_W, BASE_H = 1056, 812
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


def enable_dpi_awareness() -> None:
    """Demande à Windows de ne pas étirer la fenêtre sur les écrans mis à l'échelle."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # par écran
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class TextPainter:
    """Texte anti-aliasé (Pillow) incrusté directement dans l'image OpenCV.

    Chaque texte est rendu une fois en masque d'opacité, puis mélangé dans sa petite zone :
    aucun aller-retour complet NumPy ↔ Pillow, ce qui reste rapide même en 4K.
    """

    def __init__(self, capacity: int = 600) -> None:
        self.capacity = capacity
        self._cache: dict[tuple[str, int], tuple[np.ndarray, int, int]] = {}

    def _mask(self, text: str, font: ImageFont.ImageFont) -> tuple[np.ndarray, int, int]:
        key = (text, id(font))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        left, top, right, bottom = font.getbbox(text)
        img = Image.new("L", (max(1, right - left + 2), max(1, bottom - top + 2)))
        ImageDraw.Draw(img).text((1 - left, 1 - top), text, font=font, fill=255)
        entry = (np.asarray(img, dtype=np.float32)[..., None] / 255.0, left - 1, top - 1)
        if len(self._cache) >= self.capacity:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = entry
        return entry

    @staticmethod
    def _blend(canvas: np.ndarray, x: int, y: int, mask: np.ndarray, color) -> None:
        h, w = mask.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        a = mask[y0 - y : y1 - y, x0 - x : x1 - x]
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        mixed = roi * (1 - a) + np.asarray(color, np.float32) * a
        canvas[y0:y1, x0:x1] = mixed.astype(np.uint8)

    def draw(self, canvas, xy, text, font, color, shadow: int = 0) -> None:
        mask, ox, oy = self._mask(text, font)
        x, y = int(xy[0]) + ox, int(xy[1]) + oy
        if shadow:
            self._blend(canvas, x + shadow, y + shadow, mask, (0, 0, 0))
        self._blend(canvas, x, y, mask, color)

    @staticmethod
    def bbox(xy, text, font) -> tuple[int, int, int, int]:
        left, top, right, bottom = font.getbbox(text)
        x, y = int(xy[0]), int(xy[1])
        return x + left, y + top, x + right, y + bottom

    @staticmethod
    def width(text, font) -> float:
        return font.getlength(text)


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


@dataclass(frozen=True)
class Layout:
    """Géométrie d'une image composée de ``width`` × ``height`` pixels."""

    width: int
    height: int
    scale: float
    top: int
    bottom: int
    legend: int
    # Zone de l'image thermique, au rapport largeur/hauteur de la source (bandes sinon)
    vx: int
    vy: int
    vw: int
    vh: int

    @staticmethod
    def fit(width: int, height: int, frame_w: int = 640, frame_h: int = 480) -> Layout:
        width, height = max(width, 320), max(height, 240)
        s = min(width / BASE_W, height / BASE_H)
        top, bottom, legend = round(TOP * s), round(BOTTOM * s), round(LEGEND * s)
        area_w, area_h = width - legend, height - top - bottom
        ratio = frame_w / frame_h
        if area_w / area_h < ratio:
            vw, vh = area_w, round(area_w / ratio)
        else:
            vw, vh = round(area_h * ratio), area_h
        vx = (area_w - vw) // 2
        vy = top + (area_h - vh) // 2
        return Layout(width, height, s, top, bottom, legend, vx, vy, vw, vh)

    def px(self, value: float) -> int:
        """Longueur de référence (fenêtre 1056 × 812) convertie à cette échelle."""
        return max(1, round(value * self.scale))


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
        self._fonts: dict[int, dict[str, ImageFont.ImageFont]] = {}
        self.text = TextPainter()
        self.size = (BASE_W, BASE_H)
        self.layout = Layout.fit(*self.size)
        self.frame_shape = (480, 640)
        # Aide affichée au lancement, puis masquée automatiquement (H la rappelle).
        self.help_hide_at = time.monotonic() + 10
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
        # Scène calculée en 320 × 240 puis agrandie : identique à l'œil, 4 fois plus rapide.
        small = synthetic_heat_frame(self.step, size=(240, 320), n_blobs=3)
        return cv2.resize(small, (640, 480), interpolation=cv2.INTER_LINEAR)

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

    def fonts(self, scale: float) -> dict[str, ImageFont.ImageFont]:
        """Polices à la taille voulue (un jeu tous les 5 %, gardé en cache)."""
        key = max(8, round(scale * 20))
        if key not in self._fonts:
            k = key / 20
            self._fonts[key] = {
                "title": _font(round(19 * k), bold=True),
                "ui": _font(round(15 * k)),
                "ui_bold": _font(round(15 * k), bold=True),
                "small": _font(round(12 * k)),
                "tag": _font(round(13 * k), bold=True),
            }
        return self._fonts[key]

    def render(self, raw: np.ndarray) -> np.ndarray:
        st = self.state
        gray_raw = apply_blur(to_grayscale_intensity(raw), st.blur)
        self.frame_shape = gray_raw.shape
        h, w = gray_raw.shape
        lay = self.layout = Layout.fit(*self.size, frame_w=w, frame_h=h)
        shown = apply_contrast(gray_raw, st.contrast)
        colored = apply_colormap(shown, st.palette)
        # Agrandissement bicubique : dégradés doux, sans crénelage, même en 4K.
        view = cv2.resize(colored, (lay.vw, lay.vh), interpolation=cv2.INTER_CUBIC)

        sx, sy = lay.vw / w, lay.vh / h
        lo, hi = int(gray_raw.min()), int(gray_raw.max())
        cold_idx = np.unravel_index(int(np.argmin(gray_raw)), gray_raw.shape)
        hot_idx = np.unravel_index(int(np.argmax(gray_raw)), gray_raw.shape)

        # Bordures de couleur unie autour de l'image : fait en C par OpenCV, très rapide.
        right = lay.width - lay.vx - lay.vw
        bottom = lay.height - lay.vy - lay.vh
        canvas = cv2.copyMakeBorder(
            view, lay.vy, bottom, lay.vx, right, cv2.BORDER_CONSTANT, value=BG
        )
        self._draw_legend(canvas)

        labels: list[tuple[tuple[int, int], str, tuple[int, int, int], str]] = []
        p = lay.px
        thick = p(2)

        def to_view(y: int, x: int) -> tuple[int, int]:
            return lay.vx + int((x + 0.5) * sx), lay.vy + int((y + 0.5) * sy)

        # Point chaud et point froid
        for (y, x), color, tag in ((hot_idx, HOT, "MAX"), (cold_idx, COLD, "MIN")):
            px, py = to_view(int(y), int(x))
            cv2.circle(canvas, (px, py), p(11), (255, 255, 255), thick, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), p(8), color, thick, cv2.LINE_AA)
            temp = celsius(intensity_to_simulated_celsius(int(gray_raw[y, x])))
            labels.append(((px + p(14), py - p(10)), f"{tag} {temp}", color, "tag"))

        # Réticule central
        cx, cy = lay.vx + lay.vw // 2, lay.vy + lay.vh // 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            start = (cx + dx * p(6), cy + dy * p(6))
            end = (cx + dx * p(22), cy + dy * p(22))
            cv2.line(canvas, start, end, (255, 255, 255), thick, cv2.LINE_AA)
        center_c = intensity_to_simulated_celsius(int(gray_raw[h // 2, w // 2]))
        labels.append(((cx + p(26), cy + p(8)), celsius(center_c), (255, 255, 255), "ui_bold"))

        # Points de mesure posés au clic
        for i, (x, y) in enumerate(st.pins, start=1):
            if 0 <= x < w and 0 <= y < h:
                px, py = to_view(y, x)
                marker = cv2.MARKER_TILTED_CROSS
                cv2.drawMarker(canvas, (px, py), (255, 255, 255), marker, p(16), thick, cv2.LINE_AA)
                temp = celsius(intensity_to_simulated_celsius(int(gray_raw[y, x])))
                labels.append(((px + p(12), py + p(4)), f"P{i}  {temp}", (255, 255, 255), "tag"))

        # Fond assombri du panneau d'aide
        if st.show_help:
            x0, y0, x1, y1 = self._help_box()
            dark = cv2.convertScaleAbs(canvas[y0:y1, x0:x1], alpha=0.18)
            canvas[y0:y1, x0:x1] = cv2.add(dark, np.full_like(dark, (13, 10, 8)))

        return self._draw_text(canvas, labels, lo, hi)

    def _help_box(self) -> tuple[int, int, int, int]:
        lay = self.layout
        p = lay.px
        x0, y0 = lay.vx + p(20), lay.vy + p(20)
        x1 = min(x0 + p(390), lay.vx + lay.vw)
        y1 = min(y0 + p(18) * 2 + p(24) * (len(HELP_LINES) + 1), lay.vy + lay.vh)
        return x0, y0, x1, y1

    def _draw_legend(self, canvas: np.ndarray) -> None:
        lay = self.layout
        p = lay.px
        x0, bar_w = lay.width - lay.legend + p(34), p(22)
        top, bottom = lay.top + p(40), lay.height - lay.bottom - p(40)
        lut = palette_lut(self.state.palette).reshape(256, 3)
        ramp = lut[np.linspace(255, 0, bottom - top).astype(int)]
        canvas[top:bottom, x0 : x0 + bar_w] = ramp[:, None, :]
        cv2.rectangle(canvas, (x0 - 1, top - 1), (x0 + bar_w, bottom), (90, 90, 90), 1)

    def _draw_text(self, canvas, labels, lo: int, hi: int) -> np.ndarray:
        st = self.state
        lay = self.layout
        p = lay.px
        f = self.fonts(lay.scale)
        t = self.text

        # Barre du haut
        t.draw(canvas, (p(16), p(14)), "THERMAL CAM", f["title"], TEXT)
        title_end = p(16) + t.width("THERMAL CAM ", f["title"])
        t.draw(canvas, (title_end, p(14)), "SIM", f["title"], ACCENT)
        x = max(p(210), round(title_end + t.width("SIM", f["title"])) + p(22))
        for text, color in (
            (self.source_label, ACCENT),
            (st.palette.label, TEXT),
            (CONTRAST_LABELS[st.contrast], TEXT),
            (f"Flou {st.blur}", TEXT),
            (f"{self.fps:.0f} ips", DIM),
        ):
            box = t.bbox((x, p(17)), text, f["tag"])
            corner1 = (box[0] - p(8), box[1] - p(5))
            corner2 = (box[2] + p(8), box[3] + p(5))
            cv2.rectangle(canvas, corner1, corner2, (70, 70, 70), 1, cv2.LINE_AA)
            t.draw(canvas, (x, p(17)), text, f["tag"], color)
            x = box[2] + p(20)

        # Échelle : températures extrêmes de l'image
        lo_c = intensity_to_simulated_celsius(lo) if st.contrast != "none" else SIMULATED_MIN_C
        hi_c = intensity_to_simulated_celsius(hi) if st.contrast != "none" else SIMULATED_MAX_C
        lx = lay.width - lay.legend + p(14)
        t.draw(canvas, (lx, lay.top + p(12)), celsius(hi_c), f["ui_bold"], TEXT)
        t.draw(canvas, (lx, lay.height - lay.bottom - p(32)), celsius(lo_c), f["ui_bold"], TEXT)

        # Étiquettes des marqueurs (avec ombre pour rester lisibles sur toutes les palettes)
        for (tx, ty), text, color, font in labels:
            t.draw(canvas, (tx, ty), text, f[font], color, shadow=p(1))

        # Barre du bas : rappel honnête
        by = lay.height - lay.bottom + p(11)
        t.draw(
            canvas,
            (p(16), by),
            "Simulation : les couleurs viennent de la luminosité de l'image, "
            "pas d'une vraie mesure infrarouge.",
            f["small"],
            DIM,
        )
        hint = "H : aide  ·  F : plein écran  ·  Q : quitter"
        hint_x = lay.width - lay.legend - t.width(hint, f["small"]) - p(16)
        t.draw(canvas, (hint_x, by), hint, f["small"], DIM)

        if st.show_help:
            x0, y0, _, _ = self._help_box()
            pad, line_h = p(18), p(24)
            t.draw(canvas, (x0 + pad, y0 + pad - p(2)), "Commandes", f["ui_bold"], ACCENT)
            for i, (key, what) in enumerate(HELP_LINES, start=1):
                y = y0 + pad + i * line_h
                t.draw(canvas, (x0 + pad, y), key, f["ui_bold"], TEXT)
                t.draw(canvas, (x0 + pad + p(120), y), what, f["ui"], (200, 200, 200))

        if st.toast and time.monotonic() < st.toast_until:
            tw = t.width(st.toast, f["ui_bold"])
            x0 = int(lay.vx + (lay.vw - tw) // 2)
            y0 = lay.vy + lay.vh - p(64)
            corner2 = (int(x0 + tw + p(16)), y0 + p(30))
            cv2.rectangle(canvas, (x0 - p(16), y0 - p(10)), corner2, (28, 22, 20), -1, cv2.LINE_AA)
            t.draw(canvas, (x0, y0), st.toast, f["ui_bold"], TEXT)

        return canvas

    # ------------------------------------------------------------------ interactions

    def on_mouse(self, event: int, x: int, y: int, *_args) -> None:
        h, w = self.frame_shape
        lay = self.layout
        inside = lay.vx <= x < lay.vx + lay.vw and lay.vy <= y < lay.vy + lay.vh
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            fx, fy = int((x - lay.vx) * w / lay.vw), int((y - lay.vy) * h / lay.vh)
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
            self.help_hide_at = 0.0  # choix explicite : on ne la masque plus automatiquement
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

    def _sync_size(self) -> None:
        """Suit la taille réelle de la zone d'affichage (fenêtre redimensionnée, plein écran)."""
        try:
            _, _, ww, wh = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            return
        if ww >= 320 and wh >= 240:
            self.size = (ww, wh)

    def run(self) -> None:
        enable_dpi_awareness()
        # FREERATIO : l'image remplit la fenêtre ; comme elle est composée exactement à la
        # bonne taille, OpenCV n'a jamais à l'étirer (c'est ce qui la rendait floue).
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(WINDOW, BASE_W, BASE_H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        last = time.perf_counter()
        try:
            while True:
                frame_start = time.perf_counter()
                self._sync_size()
                if self.help_hide_at and time.monotonic() > self.help_hide_at:
                    self.state.show_help = False  # l'aide s'efface seule
                    self.help_hide_at = 0.0
                image = self.render(self.next_frame())
                cv2.imshow(WINDOW, image)
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - last, 1e-3))
                last = now
                if self.cap is None:
                    # Sans webcam, on vise ~40 images/s sans surcharger le processeur.
                    time.sleep(max(0.0, 1 / 40 - (time.perf_counter() - frame_start)))
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
