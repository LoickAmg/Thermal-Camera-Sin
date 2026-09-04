# thermal-camera-sim

Simulation de caméra thermique en Python (OpenCV + NumPy) : colormaps façon
capteur infrarouge, flou gaussien, contraste automatique, détection du point
le plus "chaud" de l'image.

## ⚠️ Ce que ce projet n'est PAS

Ce n'est **pas** un pilote pour un vrai capteur thermique (microbolomètre,
FLIR, Seek Thermal...). Un vrai capteur thermique mesure un rayonnement
infrarouge lointain (7–14 µm) totalement invisible à une webcam classique.

Ici, l'intensité lumineuse (luminance) d'une image RGB/grayscale ordinaire
est recolorée avec des palettes visuellement proches de celles utilisées en
imagerie thermique ("fer" façon FLIR, arc-en-ciel). Les "températures"
affichées sont une correspondance **linéaire et arbitraire** entre
l'intensité 0-255 et une plage 15–45 °C, choisie uniquement pour illustrer le
concept — ce n'est pas une mesure physique. Objectif : pédagogique/démo,
dans la continuité de `ascii-camera` (autre traitement d'image temps réel du
même roadmap).

## Fonctionnalités

- 3 palettes : `iron` (façon FLIR), `rainbow`, `grayscale`.
- Flou gaussien configurable (simule la résolution/le bruit plus faibles
  d'un vrai capteur thermique).
- Contraste automatique (étirement min/max) ou égalisation d'histogramme.
- Détection et repérage visuel du point le plus "chaud" de l'image (réticule
  + température simulée affichée).
- Trois sources d'entrée :
  - `--source webcam` : flux live d'une vraie webcam (fenêtre OpenCV).
  - `--source demo` : génère une ou plusieurs "sources de chaleur" mobiles de
    façon synthétique — **aucune caméra requise**, utilisé aussi par les tests.
  - `--source image` : applique le rendu à une image fixe (utile pour
    prévisualiser une palette).
- Export vidéo MP4 (`--output`), quelle que soit la source.

## Installation

```bash
pip install -e ".[test]"
```

## Utilisation

```bash
# Flux live depuis la webcam (fenêtre OpenCV, 'q' ou Échap pour quitter)
thermal-camera-sim --source webcam --colormap iron

# Démo sans webcam : exporte 5 secondes de "sources de chaleur" synthétiques
thermal-camera-sim --source demo --output demo.mp4 --frames 150 --fps 30

# Applique le rendu thermique à une image fixe
thermal-camera-sim --source image --input photo.jpg --colormap rainbow --output photo_thermal.mp4

# Désactiver le flou et forcer un contraste par égalisation d'histogramme
thermal-camera-sim --source webcam --blur 0 --contrast equalize
```

## Architecture

- `converter.py` — logique pure (colormaps, flou, contraste, détection de
  point chaud), sans aucun accès caméra : entièrement testable sans matériel
  ni fenêtre graphique.
- `camera.py` — accès webcam réelle, génération de frames synthétiques
  (mode démo), export vidéo. Couche mince au-dessus de `converter.py`.
- `main.py` — CLI (`argparse`).

## Tests

```bash
pytest -v      # 34 tests, tous sur fixtures synthétiques (pas de webcam requise)
ruff check .
```

Le mode démo (`synthetic_heat_frame`) est déterministe (fonction pure de
l'indice de frame), ce qui permet de tester tout le pipeline — y compris
l'export vidéo bout en bout — sans dépendre d'une caméra physique, exactement
comme `ascii-camera` teste sa conversion sans webcam réelle.

## Limites connues

- Pas de vrai capteur thermique : voir l'avertissement en tête de ce README.
- La fenêtre live (`--source webcam` sans `--output`) nécessite un
  environnement avec affichage graphique (X11/Wayland) — l'export vidéo
  (`--output`) fonctionne lui en tête (headless).
- Le codec `mp4v` est choisi pour sa disponibilité large ; la qualité/le
  poids ne sont pas optimisés (usage démo, pas production).

## Licence

MIT — voir [LICENSE](./LICENSE).
