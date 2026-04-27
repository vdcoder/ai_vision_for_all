from curses import meta

from PIL import Image

from ai_vision_for_all.cli import ImageMeta, draw_ruler_grid, validate_and_scale_controls


def test_validate_and_scale_controls():
    result = {"controls": [{"type": "button", "label": "OK", "x": 75, "y": 65}]}
    meta = ImageMeta(original_size=(200, 100), scaled_size=(100, 50), scale=0.5)

    out = validate_and_scale_controls(result, meta, margin=50)

    assert out["controls"][0]["scaled"] == {"x": 25, "y": 15}
    assert out["controls"][0]["original"] == {"x": 50, "y": 30}


def test_draw_ruler_grid_preserves_size():
    image = Image.new("RGB", (120, 80), "white")
    out = draw_ruler_grid(image, spacing=50, margin=50)
    assert out.size == (220, 180)
