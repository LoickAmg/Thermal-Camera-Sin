import numpy as np
import pytest

from thermal_camera_sim.viewer import BASE_H, BASE_W, Layout, ThermalViewer


def test_layout_reference_size_keeps_4_3_view():
    lay = Layout.fit(BASE_W, BASE_H)
    assert lay.scale == pytest.approx(1.0)
    assert lay.vw / lay.vh == pytest.approx(4 / 3, rel=0.01)


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440), (3840, 2160), (800, 900)])
def test_layout_fits_inside_the_window(size):
    lay = Layout.fit(*size)
    assert lay.vx >= 0 and lay.vx + lay.vw <= lay.width - lay.legend
    assert lay.vy >= lay.top and lay.vy + lay.vh <= lay.height - lay.bottom
    assert lay.vw / lay.vh == pytest.approx(4 / 3, rel=0.01)


@pytest.mark.parametrize("size", [(BASE_W, BASE_H), (1920, 1080)])
def test_render_matches_window_size(size):
    viewer = ThermalViewer(prefer_webcam=False)
    viewer.size = size
    image = viewer.render(viewer.next_frame())
    assert image.shape == (size[1], size[0], 3)
    assert image.dtype == np.uint8


def test_click_maps_back_to_source_pixels():
    viewer = ThermalViewer(prefer_webcam=False)
    viewer.size = (1920, 1080)
    viewer.render(viewer.next_frame())
    lay = viewer.layout
    viewer.on_mouse(1, lay.vx + lay.vw // 2, lay.vy + lay.vh // 2)  # EVENT_LBUTTONDOWN
    (x, y), = viewer.state.pins
    h, w = viewer.frame_shape
    assert abs(x - w // 2) <= 1 and abs(y - h // 2) <= 1
