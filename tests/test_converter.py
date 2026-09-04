import numpy as np
import pytest

from thermal_camera_sim.converter import (
    Colormap,
    apply_blur,
    apply_colormap,
    apply_contrast,
    detect_hotspot,
    draw_hotspot_marker,
    frame_to_thermal,
    intensity_to_simulated_celsius,
    stretch_contrast,
    to_grayscale_intensity,
)


def make_gradient(height: int = 20, width: int = 30) -> np.ndarray:
    """Image grayscale de test : dégradé horizontal 0 -> 255."""
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return np.tile(row, (height, 1))


def make_bright_spot(height: int = 20, width: int = 30, cx: int = 5, cy: int = 7) -> np.ndarray:
    frame = np.full((height, width), 20, dtype=np.uint8)
    frame[cy, cx] = 255
    return frame


class TestToGrayscaleIntensity:
    def test_passthrough_grayscale(self):
        gray = make_gradient()
        result = to_grayscale_intensity(gray)
        assert result.shape == gray.shape
        assert result.dtype == np.uint8

    def test_converts_bgr(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # rouge pur en BGR
        result = to_grayscale_intensity(bgr)
        assert result.shape == (10, 10)
        assert result.max() > 0

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            to_grayscale_intensity(np.zeros((10, 10, 2), dtype=np.uint8))


class TestContrast:
    def test_stretch_contrast_expands_range(self):
        narrow = np.full((10, 10), 100, dtype=np.uint8)
        narrow[0, 0] = 90
        narrow[0, 1] = 110
        stretched = stretch_contrast(narrow)
        assert stretched.min() == 0
        assert stretched.max() == 255

    def test_stretch_contrast_flat_image_no_crash(self):
        """Une image parfaitement uniforme ne doit pas provoquer de division par zéro."""
        flat = np.full((5, 5), 128, dtype=np.uint8)
        result = stretch_contrast(flat)
        assert np.array_equal(result, flat)

    def test_apply_contrast_none_is_identity(self):
        gray = make_gradient()
        assert np.array_equal(apply_contrast(gray, "none"), gray)

    def test_apply_contrast_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            apply_contrast(make_gradient(), "bogus")


class TestBlur:
    def test_blur_zero_is_identity(self):
        gray = make_bright_spot()
        assert np.array_equal(apply_blur(gray, 0), gray)

    def test_blur_smooths_bright_spot(self):
        gray = make_bright_spot()
        blurred = apply_blur(gray, 9)
        # Le pic ponctuel doit être étalé : son maximum diminue, ses voisins augmentent.
        assert blurred[7, 5] < 255
        assert blurred[7, 4] > gray[7, 4]

    def test_blur_accepts_even_kernel_without_crashing(self):
        # OpenCV exige un noyau impair : la fonction doit corriger silencieusement.
        gray = make_bright_spot()
        result = apply_blur(gray, 8)
        assert result.shape == gray.shape


class TestColormap:
    @pytest.mark.parametrize("colormap", list(Colormap))
    def test_apply_colormap_returns_bgr(self, colormap):
        gray = make_gradient()
        colored = apply_colormap(gray, colormap)
        assert colored.shape == (*gray.shape, 3)
        assert colored.dtype == np.uint8

    def test_grayscale_colormap_is_achromatic(self):
        gray = make_gradient()
        colored = apply_colormap(gray, Colormap.GRAYSCALE)
        # Les 3 canaux BGR doivent être identiques (pas de fausse couleur).
        assert np.array_equal(colored[:, :, 0], colored[:, :, 1])
        assert np.array_equal(colored[:, :, 1], colored[:, :, 2])

    def test_iron_and_rainbow_are_not_achromatic(self):
        gray = make_gradient()
        colored = apply_colormap(gray, Colormap.IRON)
        assert not np.array_equal(colored[:, :, 0], colored[:, :, 2])


class TestSimulatedTemperature:
    def test_bounds(self):
        assert intensity_to_simulated_celsius(0) == pytest.approx(15.0)
        assert intensity_to_simulated_celsius(255) == pytest.approx(45.0)

    def test_monotonic(self):
        values = [intensity_to_simulated_celsius(v) for v in range(0, 256, 17)]
        assert values == sorted(values)


class TestHotspot:
    def test_detects_brightest_pixel(self):
        gray = make_bright_spot(cx=5, cy=7)
        hotspot = detect_hotspot(gray)
        assert (hotspot.x, hotspot.y) == (5, 7)
        assert hotspot.intensity == 255
        assert hotspot.simulated_celsius == pytest.approx(45.0)

    def test_rejects_empty_image(self):
        with pytest.raises(ValueError):
            detect_hotspot(np.zeros((0, 0), dtype=np.uint8))

    def test_draw_hotspot_marker_does_not_mutate_input(self):
        gray = make_bright_spot()
        colored = apply_colormap(gray, Colormap.IRON)
        original = colored.copy()
        hotspot = detect_hotspot(gray)
        marked = draw_hotspot_marker(colored, hotspot)
        assert np.array_equal(colored, original)  # pas de mutation en place
        assert not np.array_equal(marked, colored)  # mais le résultat est bien modifié


class TestFrameToThermalPipeline:
    def test_end_to_end_on_bgr_frame(self):
        bgr = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
        result = frame_to_thermal(bgr, Colormap.IRON, "auto", blur_kernel=5)
        assert result.shape == (40, 60, 3)
        assert result.dtype == np.uint8

    def test_without_hotspot_marker(self):
        bgr = make_bright_spot(cx=5, cy=7).copy()
        bgr_3ch = np.stack([bgr, bgr, bgr], axis=-1)
        with_marker = frame_to_thermal(bgr_3ch, mark_hotspot=True)
        without_marker = frame_to_thermal(bgr_3ch, mark_hotspot=False)
        assert not np.array_equal(with_marker, without_marker)
