"""Command-line interface for AI Vision for All.

The central trick is deliberately simple: add a ruler/grid overlay to a
screenshot before sending it to a vision-capable model. The overlay gives the
model stable visual coordinates, and the returned boxes can be converted back
to the original screen coordinate system for verification or agent action.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

MAX_SIDE = 768
MARGIN = 50  # white border around the scaled screenshot, holds the ruler labels
DEFAULT_MODEL = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """\
You are a precise UI-analysis assistant that helps small AI models that do not have 
vision use digital screens. This means that the pixels you return will be used as click 
targets, so they must be accurate and harmless.

You receive an image with a screenshot, with bold red ruler labels in the margins that 
show helpful ruler guides for calculating pixel positions. Thin grid lines match the 
ruler ticks. Read the rulers like a physical ruler to locate clickable UI controls.

How to use the ruler, the human way:
- Mentally measure the horizontal distance from the pixel you want to precisly locate to the nearest vertical grid line to the left of this pixel, remember this value, then follow the vertical grid line to the top ruler's label in the margin to get the lines's x value, add that label to the remembered measured value to get the final x coordinate.
- Do the same for y, measure the vertical distance to the nearest horizontal grid line above this pixel, remember this value, then follow the horizontal grid line to the left ruler's label in the margin to get the line's y value, add that label to the remembered measured value to get the final y coordinate.

Your tasks:
1. Describe what is visible on screen in the context of the goal.
2. List the potentially clickable UI controls relevant to the goal in a
   precise manner: buttons, links, inputs, menus, tabs, checkboxes, icons,
   selectable rows, etc.

For each control return the best clickable point coordinates: a single
best-chance pixel. No rectangles, regions, sizes or bounds.

For each control return:
  - type: control kind, such as button, link, input, menu_item, tab, icon_button
  - label: visible text, or a short description when there is no text
  - x: x pixel of the click target (use the red rulers)
  - y: y pixel of the click target (use the red rulers)
  - reason: how was the click point determined, what makes this point the
    best choice, and why is this control relevant to the goal?
  - confidence: number from 0.0 to 1.0

Hints on choosing the click pixel:
  - buttons, links, tabs, menu items: on the center of the visible glyph or label text
  - text inputs and search fields: inside the center of the editable area
  - dropdowns and combo boxes: on the center of the chevron/arrow if visible, otherwise
    on the center of the label
  - checkboxes and radio buttons: on the center of the box or radio circle
  - icons: on the center of the icon glyph itself
  - selectable rows: on the center of the most distinctive text or icon in the row > based on the apparent row boundaries

All coordinates must be integers and must fall inside the screenshot
rectangle whose bounds are given in the user prompt. Return valid JSON only:
{
  "description": "...",
  "controls": [
    {"type":"button","label":"...","x":<center>,"y":<center>,"reason":"...","confidence":0.0}
  ]
}
"""

_MEDIA_RESOLUTION = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}


@dataclass(frozen=True)
class ImageMeta:
    original_size: tuple[int, int]
    scaled_size: tuple[int, int]
    scale: float


def parse_rectangle(spec: str) -> tuple[int, int, int, int]:
    """Parse a 'x,y,w,h' rectangle in original-image coordinates.

    Whitespace tolerant. Raises argparse.ArgumentTypeError on malformed input.
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--rectangle expects 'x,y,w,h' (got {spec!r})"
        )
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--rectangle values must be integers (got {spec!r})"
        ) from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            f"--rectangle width and height must be positive (got {w}x{h})"
        )
    return x, y, w, h


def load_and_scale(path: Path, max_side: int = MAX_SIDE) -> tuple[Image.Image, ImageMeta]:
    """Load an image and scale it so the longest side is at most max_side."""
    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size
    longest = max(orig_w, orig_h)
    if longest <= max_side:
        return image, ImageMeta((orig_w, orig_h), (orig_w, orig_h), 1.0)

    scale = max_side / longest
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return scaled, ImageMeta((orig_w, orig_h), (new_w, new_h), scale)


def _load_label_font(size: int = 14) -> ImageFont.ImageFont:
    """Return a bold sans-serif font; fall back to PIL's default if none found."""
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_ruler_grid(image: Image.Image, spacing: int = 50, margin: int = MARGIN) -> Image.Image:
    """Place *image* on a white canvas with rulers labelled in canvas coordinates.

    The screenshot is pasted at offset (margin, margin) on a white canvas of
    size (w + 2*margin) x (h + 2*margin). Grid lines and ruler labels both use
    the canvas pixel coordinate system, so the top-left grid line is at
    canvas (margin, margin) and the first label reads ``margin`` rather than 0.
    With margin == spacing, the ruler ticks fall exactly on the grid lines.

    The model is told to predict click pixels in this canvas coordinate space;
    `validate_and_scale_controls` later subtracts the margin and unscales to
    recover original-screenshot coordinates.
    """
    import numpy as np

    inner_arr = np.array(image)
    h, w = inner_arr.shape[:2]

    # XOR grid lines on the inner image (skip the very edge so the screenshot
    # boundary stays clean against the white margin).
    for y in range(spacing, h, spacing):
        inner_arr[y, :] = inner_arr[y, :] ^ 0xFF
    for x in range(spacing, w, spacing):
        inner_arr[:, x] = inner_arr[:, x] ^ 0xFF

    inner_img = Image.fromarray(inner_arr)

    canvas_w = w + 2 * margin
    canvas_h = h + 2 * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(inner_img, (margin, margin))

    draw = ImageDraw.Draw(canvas)
    font = _load_label_font(14)

    def text_size(text: str) -> tuple[int, int]:
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        except AttributeError:
            print("Warning: textbbox not available, using rough text size estimates.", file=sys.stderr)
            return len(text) * 7, 12

    # X-axis ruler labels: every grid line, in CANVAS coordinates (start at margin).
    cx_start = margin
    cx_end = margin + w  # exclusive upper edge
    for cx in range(cx_start, cx_end, spacing):
        label = str(cx)
        tw, th = text_size(label)
        lx = cx - tw // 2
        draw.text((lx, max(1, margin - th - 6)), label, fill="red", font=font)
        draw.text((lx, margin + h + 2), label, fill="red", font=font)

    # Y-axis ruler labels: every grid line, in CANVAS coordinates (start at margin).
    cy_start = margin
    cy_end = margin + h
    for cy in range(cy_start, cy_end, spacing):
        label = str(cy)
        tw, th = text_size(label)
        ly = cy - th // 2
        draw.text((margin - tw - 4, ly), label, fill="red", font=font)
        draw.text((margin + w + 4, ly), label, fill="red", font=font)

    return canvas


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a model response."""
    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def validate_and_scale_controls(
    result: dict[str, Any],
    meta: ImageMeta,
    margin: int = MARGIN,
    crop_offset: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Convert canvas-space click points back to original-image pixels.

    The model predicts (x, y) in the canvas coordinate system used by the
    rulers, which starts at (margin, margin) for the top-left of the scaled
    screenshot. We:
      1. subtract the margin to get inner-image (scaled) coordinates,
      2. clamp to the scaled image bounds,
      3. divide by `meta.scale` to recover (cropped) original coordinates,
      4. add `crop_offset` to recover full-original-image coordinates when the
         analysis ran on a sub-rectangle of the source.
    """
    sw, sh = meta.scaled_size
    scale = meta.scale
    ox, oy = crop_offset
    controls = result.get("controls") or []

    for control in controls:
        cx = int(round(float(control.get("x", margin))))
        cy = int(round(float(control.get("y", margin))))

        # Canvas -> inner (scaled) coordinates.
        ix = cx - margin
        iy = cy - margin

        # Clamp to the scaled screenshot bounds.
        ix = max(0, min(ix, sw - 1))
        iy = max(0, min(iy, sh - 1))

        # Drop any legacy rectangle keys the model may emit out of habit.
        for key in ("width", "height", "interact_x", "interact_y", "center", "interact", "interact_source"):
            control.pop(key, None)

        # Keep the canvas-space prediction the model returned (post-clamp) for
        # debugging, plus the inner (scaled) and full-original-image mappings.
        control["x"] = ix + margin
        control["y"] = iy + margin
        control["scaled"] = {"x": ix, "y": iy}
        control["original"] = {
            "x": round(ix / scale) + ox,
            "y": round(iy / scale) + oy,
        }

    result["controls"] = controls
    return result


def call_gemini(
    image: Image.Image,
    goal: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    media_resolution: str = "medium",
    thinking_budget: int = 0,
    grid_spacing: int = 50,
    inner_size: tuple[int, int] | None = None,
    margin: int = MARGIN,
) -> dict[str, Any]:
    """Analyze a ruler-marked screenshot with Gemini."""
    client = genai.Client(api_key=api_key)
    iw, ih = inner_size if inner_size else image.size
    canvas_w, canvas_h = iw + 2 * margin, ih + 2 * margin
    x_min, x_max = margin, margin + iw - 1
    y_min, y_max = margin, margin + ih - 1
    print(f"Canvas size: {canvas_w}x{canvas_h}. Ruler x range: {x_min}-{x_max}, y range: {y_min}-{y_max}.", file=sys.stderr)
    prompt = (
        f"Valid x range: {x_min}-{x_max}. Valid y range: {y_min}-{y_max}.\n"
        f"Goal: {goal}"
    )
    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0,
    }
    if media_resolution in _MEDIA_RESOLUTION:
        config_kwargs["media_resolution"] = _MEDIA_RESOLUTION[media_resolution]
    if thinking_budget > 0:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(**config_kwargs),
    )

    parsed = extract_json(response.text.strip())
    if parsed is None:
        return {"error": "JSON parse failed", "raw_response": response.text.strip()}

    usage = {}
    if response.usage_metadata:
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "candidates_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }
        if response.usage_metadata.thoughts_token_count:
            usage["thoughts_tokens"] = response.usage_metadata.thoughts_token_count
    parsed["token_usage"] = usage
    return parsed


def draw_overlay(original_image_path: Path, analysis_path: Path, output_path: Path) -> None:
    """Draw click points as labelled circles on the original screenshot."""
    image = Image.open(original_image_path).convert("RGB")
    with analysis_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    draw = ImageDraw.Draw(image)
    palette = {
        "button": "lime",
        "input": "cyan",
        "text_input": "cyan",
        "input_box": "cyan",
        "link": "orange",
        "menu_item": "yellow",
        "tab": "magenta",
        "checkbox": "red",
        "icon_button": "white",
    }
    radius = 12

    for control in data.get("controls", []):
        point = control.get("original", control)
        x = int(point["x"])
        y = int(point["y"])
        color = palette.get(control.get("type"), "yellow")
        # Outer black halo for visibility on light backgrounds.
        draw.ellipse([x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1], outline="black", width=3)
        # Coloured ring at the click pixel.
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=2)
        # Crosshair at the exact pixel.
        draw.line([x - 6, y, x + 6, y], fill="red", width=2)
        draw.line([x, y - 6, x, y + 6], fill="red", width=2)
        draw.text((x + radius + 4, y - 6), f"{control.get('label', '?')} ({x},{y})", fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def command_preview(args: argparse.Namespace) -> int:
    inner_max = max(1, args.max_side - 2 * args.margin)
    image, _meta = load_and_scale(Path(args.image), inner_max)
    preview = draw_ruler_grid(image, args.grid, args.margin)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.output)
    print(f"Saved preview -> {args.output}")
    return 0


def _select_best_control(controls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the highest-confidence control with a usable `original` point."""
    best: dict[str, Any] | None = None
    best_conf = -1.0
    for control in controls:
        original = control.get("original")
        if not isinstance(original, dict) or "x" not in original or "y" not in original:
            continue
        try:
            conf = float(control.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf > best_conf:
            best_conf = conf
            best = control
    return best


def _run_single_pass(
    *,
    full_image: Image.Image,
    full_size: tuple[int, int],
    rectangle: tuple[int, int, int, int] | None,
    goal_with_target: str,
    target: str | None,
    api_key: str,
    args: argparse.Namespace,
    preview_path: Path | None,
) -> dict[str, Any] | None:
    """Run one analyze pass and return its full output dict (or None on error)."""
    full_w, full_h = full_size
    crop_offset = (0, 0)
    crop_meta: dict[str, Any] | None = None

    if rectangle:
        rx, ry, rw, rh = rectangle
        x0 = max(0, min(rx, full_w))
        y0 = max(0, min(ry, full_h))
        x1 = max(0, min(rx + rw, full_w))
        y1 = max(0, min(ry + rh, full_h))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            print(
                f"--rectangle {rectangle} does not overlap the {full_w}x{full_h} image.",
                file=sys.stderr,
            )
            return None
        source_image = full_image.crop((x0, y0, x1, y1))
        crop_offset = (x0, y0)
        crop_meta = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
    else:
        source_image = full_image

    inner_max = max(1, args.max_side - 2 * args.margin)
    src_w, src_h = source_image.size
    longest = max(src_w, src_h)
    if longest <= inner_max:
        scaled = source_image
        scaled_size = (src_w, src_h)
        scale = 1.0
    else:
        scale = inner_max / longest
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        scaled = source_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        scaled_size = (new_w, new_h)
    meta = ImageMeta(original_size=(src_w, src_h), scaled_size=scaled_size, scale=scale)

    marked = draw_ruler_grid(scaled, args.grid, args.margin)
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        marked.save(preview_path)

    result = call_gemini(
        marked,
        goal=goal_with_target,
        api_key=api_key,
        model=args.model,
        media_resolution=args.media_res,
        thinking_budget=args.think,
        grid_spacing=args.grid,
        inner_size=meta.scaled_size,
        margin=args.margin,
    )
    result = validate_and_scale_controls(result, meta, args.margin, crop_offset)
    return {
        "meta": {
            "image_size": [full_w, full_h],
            "crop": crop_meta,
            "target": target,
            "goal": goal_with_target,
            "source_size": list(meta.original_size),
            "scaled_size": list(meta.scaled_size),
            "scale": round(meta.scale, 6),
            "margin": args.margin,
            "model": args.model,
            "media_resolution": args.media_res,
            "grid_spacing": args.grid,
        },
        **result,
    }


def _pass_output_path(base: Path, pass_index: int, total_passes: int) -> Path:
    """Return `base.passK<ext>` (1-based). Single-pass returns base unchanged."""
    if total_passes <= 1:
        return base
    return base.with_name(f"{base.stem}.pass{pass_index}{base.suffix}")


def command_analyze(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY is not set. Copy .env.example to .env or export it.", file=sys.stderr)
        return 2

    if args.passes < 1:
        print("--passes must be >= 1", file=sys.stderr)
        return 2
    if not (0.0 < args.crop_factor <= 1.0):
        print("--crop-factor must be in (0.0, 1.0]", file=sys.stderr)
        return 2

    full_image = Image.open(Path(args.image)).convert("RGB")
    full_w, full_h = full_image.size

    goal_with_target = args.goal
    if args.target:
        goal_with_target = f"{args.goal} now trying to click on: {args.target}"

    output_base = Path(args.output) if args.output else None
    preview_base = Path(args.preview) if args.preview else None

    # First pass uses --rectangle if provided, otherwise the full image.
    next_rect: tuple[int, int, int, int] | None = args.rectangle

    last_output: dict[str, Any] | None = None
    last_path: Path | None = None

    for pass_index in range(1, args.passes + 1):
        pass_preview = _pass_output_path(preview_base, pass_index, args.passes) if preview_base else None

        pass_output = _run_single_pass(
            full_image=full_image,
            full_size=(full_w, full_h),
            rectangle=next_rect,
            goal_with_target=goal_with_target,
            target=args.target,
            api_key=api_key,
            args=args,
            preview_path=pass_preview,
        )
        if pass_output is None:
            return 2

        # Annotate this pass with multi-pass bookkeeping.
        pass_output["meta"]["pass"] = pass_index
        pass_output["meta"]["passes_total"] = args.passes
        pass_output["meta"]["crop_factor"] = args.crop_factor

        text = json.dumps(pass_output, indent=2, ensure_ascii=False)
        if output_base:
            pass_path = _pass_output_path(output_base, pass_index, args.passes)
            pass_path.parent.mkdir(parents=True, exist_ok=True)
            pass_path.write_text(text, encoding="utf-8")
            print(f"Saved analysis (pass {pass_index}/{args.passes}) -> {pass_path}", file=sys.stderr)
            last_path = pass_path
        elif pass_index == args.passes:
            # No --output: print only the final pass to stdout.
            print(text)
        last_output = pass_output

        # Compute the next pass's rectangle by zooming around the best control.
        if pass_index < args.passes:
            best = _select_best_control(pass_output.get("controls") or [])
            if not best:
                print(
                    f"Pass {pass_index} returned no usable controls; stopping early.",
                    file=sys.stderr,
                )
                break
            cx = int(best["original"]["x"])
            cy = int(best["original"]["y"])
            # Each axis shrinks by crop_factor relative to the *previous* source size
            # so successive passes progressively zoom in.
            prev_src_w, prev_src_h = pass_output["meta"]["source_size"]
            new_w = max(1, int(round(prev_src_w * args.crop_factor)))
            new_h = max(1, int(round(prev_src_h * args.crop_factor)))
            new_x = cx - new_w // 2
            new_y = cy - new_h // 2
            next_rect = (new_x, new_y, new_w, new_h)

    # Also copy the final pass to the base --output / --preview paths so callers
    # that don't care about the per-pass chain get their expected files.
    if args.passes > 1:
        if output_base and last_output is not None and last_path != output_base:
            output_base.parent.mkdir(parents=True, exist_ok=True)
            output_base.write_text(json.dumps(last_output, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved final analysis -> {output_base}", file=sys.stderr)
        if preview_base:
            final_preview = _pass_output_path(preview_base, args.passes, args.passes)
            if final_preview != preview_base and final_preview.exists():
                import shutil
                shutil.copy2(final_preview, preview_base)
                print(f"Saved final preview -> {preview_base}", file=sys.stderr)

    return 0


def command_overlay(args: argparse.Namespace) -> int:
    draw_overlay(Path(args.image), Path(args.analysis), Path(args.output))
    print(f"Saved overlay -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aivision", description="Ruler-guided AI screen vision for GUI workflows.")
    sub = parser.add_subparsers(required=True)

    preview = sub.add_parser("preview", help="Create a ruler-grid preview image without calling a model.")
    preview.add_argument("image")
    preview.add_argument("--output", "-o", default="outputs/ruler_preview.png")
    preview.add_argument("--grid", type=int, default=50)
    preview.add_argument("--max-side", type=int, default=MAX_SIDE)
    preview.add_argument("--margin", type=int, default=MARGIN)
    preview.set_defaults(func=command_preview)

    analyze = sub.add_parser("analyze", help="Analyze a screenshot and return structured controls.")
    analyze.add_argument("image")
    analyze.add_argument("--goal", "-g", required=True)
    analyze.add_argument("--output", "-o")
    analyze.add_argument("--preview")
    analyze.add_argument("--grid", type=int, default=50)
    analyze.add_argument("--max-side", type=int, default=MAX_SIDE)
    analyze.add_argument("--margin", type=int, default=MARGIN)
    analyze.add_argument("--model", default=DEFAULT_MODEL)
    analyze.add_argument("--media-res", choices=sorted(_MEDIA_RESOLUTION), default="medium")
    analyze.add_argument("--think", type=int, default=0)
    analyze.add_argument(
        "--target",
        help="Optional second-pass refinement hint, e.g. 'the Wi-Fi toggle'. "
             "Appended to the goal so the model focuses on a single control.",
    )
    analyze.add_argument(
        "--rectangle",
        type=parse_rectangle,
        metavar="X,Y,W,H",
        help="Crop the source to this rectangle (in original-image pixels) "
             "before analysis. Returned coordinates remain in the full "
             "original-image coordinate system.",
    )
    analyze.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Run N analysis passes; each pass after the first crops around "
             "the previous pass's best click point. Per-pass JSONs are saved "
             "as <output>.passK<ext>.",
    )
    analyze.add_argument(
        "--crop-factor",
        type=float,
        default=0.5,
        metavar="F",
        help="Diameter ratio (0..1] used to size each follow-up pass's crop "
             "relative to the previous source. Default 0.5: each axis halves.",
    )
    analyze.set_defaults(func=command_analyze)

    overlay = sub.add_parser("overlay", help="Draw analysis boxes on the original screenshot.")
    overlay.add_argument("image")
    overlay.add_argument("analysis")
    overlay.add_argument("--output", "-o", default="outputs/verify_overlay.png")
    overlay.set_defaults(func=command_overlay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
