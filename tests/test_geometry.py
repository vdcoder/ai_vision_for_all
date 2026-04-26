from pathlib import Path

from PIL import Image

from ai_vision_for_all.cli import ImageMeta, draw_ruler_grid, validate_and_scale_controls


def test_validate_and_scale_controls_adds_original_coordinates():
    result = {
        "controls": [
            {"type": "button", "label": "OK", "x": 50, "y": 25, "width": 20, "height": 10}
        ]
    }
    meta = ImageMeta(original_size=(200, 100), scaled_size=(100, 50), scale=0.5)
    out = validate_and_scale_controls(result, meta)
    control = out["controls"][0]

    assert control["center"] == [60, 30]
    assert control["original"]["x"] == 100
    assert control["original"]["y"] == 50
    assert control["original"]["center"] == [120, 60]


def test_validate_and_scale_controls_clamps_bounds():
    result = {
        "controls": [
            {"type": "button", "label": "OK", "x": 95, "y": 45, "width": 99, "height": 99}
        ]
    }
    meta = ImageMeta(original_size=(200, 100), scaled_size=(100, 50), scale=0.5)
    out = validate_and_scale_controls(result, meta)
    control = out["controls"][0]

    assert control["x"] == 95
    assert control["y"] == 45
    assert control["width"] == 5
    assert control["height"] == 5


def test_draw_ruler_grid_preserves_size():
    image = Image.new("RGB", (120, 80), "white")
    out = draw_ruler_grid(image, spacing=50)
    assert out.size == (120, 80)
