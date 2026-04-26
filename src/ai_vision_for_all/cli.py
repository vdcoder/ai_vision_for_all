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
from PIL import Image, ImageDraw

MAX_SIDE = 768
DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """\
You are a precise UI-analysis assistant.

You receive a screenshot with ruler markings along the top and left edges,
plus thin grid lines crossing the image. The rulers show pixel positions in
the image you are looking at. Read them like a physical ruler to locate UI
elements.

Your tasks:
1. Describe what is visible on screen in the context of the goal.
2. List the interactive UI controls relevant to the goal: buttons, links,
   inputs, menus, tabs, checkboxes, icons, selectable rows, etc.

For each control return:
  - type: control kind, such as button, link, input, menu_item, tab, icon_button
  - label: visible text, or a short description when there is no text
  - x: left edge in pixels, read from the top ruler
  - y: top edge in pixels, read from the left ruler
  - width: box width in pixels
  - height: box height in pixels
  - confidence: number from 0.0 to 1.0

All coordinates must be integers in the pixel space of this image, not the
original full-resolution screenshot. Return valid JSON only:
{
  "description": "...",
  "controls": [
    {"type":"button","label":"...","x":0,"y":0,"width":0,"height":0,"confidence":0.0}
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


def draw_ruler_grid(image: Image.Image, spacing: int = 50) -> Image.Image:
    """Draw XOR grid lines with ruler labels on all four image edges."""
    import numpy as np

    arr = np.array(image)
    h, w = arr.shape[:2]

    for y in range(0, h, spacing):
        arr[y, :] = arr[y, :] ^ 0xFF
    for x in range(0, w, spacing):
        arr[:, x] = arr[:, x] ^ 0xFF

    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)

    def outlined_text(x: int, y: int, text: str) -> None:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, fill="black")
        draw.text((x, y), text, fill="white")

    for x in range(0, w, spacing):
        label = str(x)
        outlined_text(x + 2, 1, label)
        outlined_text(x + 2, max(0, h - 12), label)

    for y in range(0, h, spacing):
        label = str(y)
        outlined_text(1, y + 2, label)
        outlined_text(max(0, w - 2 - len(label) * 6), y + 2, label)

    return out


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


def validate_and_scale_controls(result: dict[str, Any], meta: ImageMeta) -> dict[str, Any]:
    """Clamp scaled-image boxes and add original-image coordinates."""
    sw, sh = meta.scaled_size
    scale = meta.scale
    controls = result.get("controls") or []

    for control in controls:
        x = int(round(float(control.get("x", 0))))
        y = int(round(float(control.get("y", 0))))
        width = int(round(float(control.get("width", 0))))
        height = int(round(float(control.get("height", 0))))

        x = max(0, min(x, sw - 1))
        y = max(0, min(y, sh - 1))
        width = max(0, min(width, sw - x))
        height = max(0, min(height, sh - y))

        control["x"] = x
        control["y"] = y
        control["width"] = width
        control["height"] = height
        control["center"] = [x + width // 2, y + height // 2]
        control["original"] = {
            "x": round(x / scale),
            "y": round(y / scale),
            "width": round(width / scale),
            "height": round(height / scale),
            "center": [round((x + width / 2) / scale), round((y + height / 2) / scale)],
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
) -> dict[str, Any]:
    """Analyze a ruler-marked screenshot with Gemini."""
    client = genai.Client(api_key=api_key)
    w, h = image.size
    prompt = (
        "Ruler markings along the top (x) and left (y) edges show "
        f"pixel positions, with grid lines every {grid_spacing} pixels. "
        "Read the ruler numbers and follow the grid lines to "
        "determine bounding box coordinates. "
        "ALL coordinates must fall within the image dimensions above.\n"
        f"This image is {w}x{h} pixels. "
        f"Valid x range: 0-{w - 1}. Valid y range: 0-{h - 1}.\n"
        f"Goal: {goal}"
    )
    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.2,
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
    """Draw verified boxes on the original screenshot."""
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

    for control in data.get("controls", []):
        box = control.get("original", control)
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["width"])
        h = int(box["height"])
        cx = x + w // 2
        cy = y + h // 2
        color = palette.get(control.get("type"), "yellow")
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.line([cx - 8, cy, cx + 8, cy], fill="red", width=2)
        draw.line([cx, cy - 8, cx, cy + 8], fill="red", width=2)
        draw.text((x, max(0, y - 14)), f"{control.get('label', '?')} ({cx},{cy})", fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def command_preview(args: argparse.Namespace) -> int:
    image, _meta = load_and_scale(Path(args.image), args.max_side)
    preview = draw_ruler_grid(image, args.grid)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.output)
    print(f"Saved preview -> {args.output}")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY is not set. Copy .env.example to .env or export it.", file=sys.stderr)
        return 2

    image, meta = load_and_scale(Path(args.image), args.max_side)
    marked = draw_ruler_grid(image, args.grid)
    if args.preview:
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        marked.save(args.preview)

    result = call_gemini(
        marked,
        goal=args.goal,
        api_key=api_key,
        model=args.model,
        media_resolution=args.media_res,
        thinking_budget=args.think,
        grid_spacing=args.grid,
    )
    result = validate_and_scale_controls(result, meta)
    output = {
        "meta": {
            "original_size": list(meta.original_size),
            "scaled_size": list(meta.scaled_size),
            "scale": round(meta.scale, 6),
            "model": args.model,
            "media_resolution": args.media_res,
            "grid_spacing": args.grid,
        },
        **result,
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Saved analysis -> {args.output}", file=sys.stderr)
    else:
        print(text)
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
    preview.set_defaults(func=command_preview)

    analyze = sub.add_parser("analyze", help="Analyze a screenshot and return structured controls.")
    analyze.add_argument("image")
    analyze.add_argument("--goal", "-g", required=True)
    analyze.add_argument("--output", "-o")
    analyze.add_argument("--preview")
    analyze.add_argument("--grid", type=int, default=50)
    analyze.add_argument("--max-side", type=int, default=MAX_SIDE)
    analyze.add_argument("--model", default=DEFAULT_MODEL)
    analyze.add_argument("--media-res", choices=sorted(_MEDIA_RESOLUTION), default="medium")
    analyze.add_argument("--think", type=int, default=0)
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
