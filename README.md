# AI Vision for All

**Ruler-guided screen vision for AI agents that need to understand and operate GUIs.**

Most agent systems assume the model already has reliable screen vision. Many useful agents do not. This project gives any orchestrator a practical bridge: take a screenshot, draw a machine-readable ruler/grid over it, ask a vision model to identify controls, then convert the result back into original screen coordinates for verification or GUI automation.

The core idea is intentionally small:

1. Capture or provide a screenshot.
2. Downscale it to a predictable vision-token budget.
3. Add visual coordinate rulers and grid lines.
4. Ask a vision model for structured UI controls.
5. Validate/clamp the boxes.
6. Optionally draw an overlay on the original screenshot before any click is trusted.

This is useful for:

- agentic GUI workflows,
- “computer use” experiments,
- AI systems that can reason but lack native vision,
- cross-agent delegation where one AI acts as the visual observer for another,
- accessibility-style UI mapping,
- deterministic verification before automation.

## Why the ruler matters

Vision models can describe screens well, but raw pixel coordinates are often unreliable. The ruler overlay gives the model visible spatial anchors. Instead of guessing where a button is, the model can read grid/ruler labels and return coordinates in the image’s actual pixel space.

This project treats the model as an **observer**, not an oracle. The returned coordinates are meant to be inspected, clamped, verified, and only then handed to automation code.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

## Quick start

Create a ruler preview without calling any model:

```bash
aivision preview screenshot.png --output outputs/ruler_preview.png
```

Analyze a screenshot:

```bash
aivision analyze screenshot.png \
  --goal "create a new C++ project" \
  --preview outputs/ruler_preview.png \
  --output outputs/result.json
```

Draw the model’s boxes back onto the original full-resolution screenshot:

```bash
aivision overlay screenshot.png outputs/result.json \
  --output outputs/verify_overlay.png
```

The JSON includes scaled-image coordinates plus original-image coordinates:

```json
{
  "controls": [
    {
      "type": "link",
      "label": "Create a new project",
      "x": 585,
      "y": 100,
      "width": 140,
      "height": 20,
      "center": [655, 110],
      "original": {
        "x": 1075,
        "y": 184,
        "width": 257,
        "height": 37,
        "center": [1203, 202]
      }
    }
  ]
}
```

## Current status

This is an early developer prototype. The included CLI is enough to demonstrate the concept and produce inspectable JSON/overlays. It is not yet a safe autonomous desktop driver.

Recommended next steps before using it for real automation:

- add OS-level screenshot capture helpers,
- add optional mouse/keyboard action adapters behind explicit confirmation gates,
- add OCR fallback for text-heavy UIs,
- add regression tests using fixed screenshots,
- add confidence thresholds and a “verify before click” policy,
- add a local-model path for privacy-sensitive screens.

## Safety posture

Do not run automation against destructive UIs without a confirmation layer. For agentic workflows, prefer this chain:

```text
observe -> propose action -> verify target -> require policy approval -> execute -> observe again
```

The project is designed to make that verification loop easier, not to bypass it.

## License

MIT
