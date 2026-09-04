import numpy as np

from thermal_camera_sim.camera import (
    DEFAULT_DEMO_SIZE,
    export_thermal_video,
    iter_synthetic_frames,
    load_image_frame,
    synthetic_heat_frame,
)
from thermal_camera_sim.converter import Colormap


class TestSyntheticHeatFrame:
    def test_shape_and_dtype(self):
        frame = synthetic_heat_frame(step=0, size=(20, 30))
        assert frame.shape == (20, 30)
        assert frame.dtype == np.uint8

    def test_deterministic_given_step(self):
        a = synthetic_heat_frame(step=5, size=(20, 30))
        b = synthetic_heat_frame(step=5, size=(20, 30))
        assert np.array_equal(a, b)

    def test_changes_over_steps(self):
        a = synthetic_heat_frame(step=0, size=(20, 30))
        b = synthetic_heat_frame(step=10, size=(20, 30))
        assert not np.array_equal(a, b)

    def test_default_size_used_when_unspecified(self):
        frame = synthetic_heat_frame(step=0)
        assert frame.shape == DEFAULT_DEMO_SIZE

    def test_iter_synthetic_frames_yields_n_frames(self):
        frames = list(iter_synthetic_frames(4, size=(10, 10)))
        assert len(frames) == 4
        assert all(f.shape == (10, 10) for f in frames)


class TestLoadImageFrame:
    def test_missing_file_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_image_frame(tmp_path / "does-not-exist.png")

    def test_loads_real_image(self, tmp_path):
        import cv2

        img_path = tmp_path / "fixture.png"
        cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
        frame = load_image_frame(img_path)
        assert frame.shape == (10, 10, 3)


class TestExportThermalVideo:
    def test_export_demo_creates_playable_file(self, tmp_path):
        import cv2

        output = tmp_path / "out.mp4"
        result_path = export_thermal_video(output_path=output, source="demo", n_frames=5, fps=10)
        assert result_path == output
        assert output.exists()
        assert output.stat().st_size > 0

        cap = cv2.VideoCapture(str(output))
        assert cap.isOpened()
        frame_count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frame_count += 1
        cap.release()
        assert frame_count == 5

    def test_export_image_source_loops_still_frame(self, tmp_path):
        import cv2

        img_path = tmp_path / "still.png"
        cv2.imwrite(str(img_path), np.full((16, 16, 3), 200, dtype=np.uint8))
        output = tmp_path / "out.mp4"
        export_thermal_video(
            output_path=output,
            source="image",
            input_path=img_path,
            n_frames=3,
            fps=10,
        )
        assert output.exists() and output.stat().st_size > 0

    def test_export_image_source_without_input_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            export_thermal_video(output_path=tmp_path / "out.mp4", source="image", input_path=None)

    def test_export_unknown_source_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            export_thermal_video(output_path=tmp_path / "out.mp4", source="bogus")

    def test_export_respects_colormap_choice(self, tmp_path):
        """Deux colormaps différentes doivent produire des fichiers de contenu différent."""
        out_iron = tmp_path / "iron.mp4"
        out_gray = tmp_path / "gray.mp4"
        export_thermal_video(out_iron, source="demo", n_frames=3, colormap=Colormap.IRON)
        export_thermal_video(out_gray, source="demo", n_frames=3, colormap=Colormap.GRAYSCALE)
        assert out_iron.read_bytes() != out_gray.read_bytes()
