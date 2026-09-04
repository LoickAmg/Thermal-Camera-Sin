"""thermal-camera-sim — simulation de caméra thermique à partir d'une webcam.

Avertissement honnête : ceci n'est PAS un pilote pour un vrai capteur thermique
(microbolomètre, FLIR, etc.). Il recolore l'intensité lumineuse d'une image
RGB/grayscale ordinaire selon des palettes "thermiques" classiques (fer,
arc-en-ciel), pour un usage pédagogique/démo. Les "températures" affichées
sont une correspondance linéaire arbitraire avec l'intensité, pas une mesure
physique réelle.
"""

__version__ = "0.1.0"
