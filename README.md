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

## En bref : à quoi ça sert ?

Donner à n'importe quelle webcam le **look d'une caméra thermique** : l'image est recolorée
avec les palettes des vraies caméras (Iron façon FLIR, Rainbow, White/Black hot, Arctic, Lava),
et l'interface affiche ce qu'on voit sur ces appareils — échelle de « températures », réticule
central, points le plus chaud et le plus froid, points de mesure posés à la souris. Les
« températures » sont tirées de la **luminosité** de l'image (pas d'un capteur infrarouge) :
c'est un outil pédagogique et ludique pour comprendre le traitement d'image, pas un instrument.

## Lancer l'application (sans commande)

Double-cliquez sur `dist/ThermalCameraSim.exe` (après l'avoir construit, voir plus bas) :
la fenêtre s'ouvre sur la webcam, ou sur une scène de démonstration animée s'il n'y en a pas.

| Touche | Action |
|---|---|
| `P` ou `1`–`6` | changer de palette |
| `C` | contraste : auto / égalisé / brut |
| `+` / `−` | flou du « capteur » |
| clic gauche / droit | poser (3 max) / effacer des points de mesure |
| `W` | basculer webcam / démo |
| `S` | capture PNG dans `Images/Thermal Camera Sim` |
| `F`, `H`, `Q` | plein écran, aide, quitter |

Construire l'exécutable (Windows) :

```bash
pip install -e ".[build]"
python scripts/build_exe.py      # → dist/ThermalCameraSim.exe (autonome, sans console)
```

## Fonctionnalités
- 6 palettes : `iron` (façon FLIR, dessinée à la main), `rainbow`, `grayscale` (white hot),
  `black-hot`, `arctic`, `lava`.
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
# Fenêtre interactive : webcam si disponible, sinon démo (c'est aussi ce que fait l'exécutable)
thermal-camera-sim

# Forcer la démo en direct, ou une image fixe
thermal-camera-sim --source demo
thermal-camera-sim --source image --input photo.jpg

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
- `viewer.py` — fenêtre interactive (HUD, échelle, mesures, raccourcis), texte rendu avec Pillow.
- `main.py` — CLI (`argparse`) ; sans `--output`, ouvre la fenêtre interactive.

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
