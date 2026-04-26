# Architecture

## Concept

AI Vision for All is a visual adapter for agent systems. It turns a normal screenshot into a coordinate-readable screenshot, asks a vision model to extract UI controls, and returns structured JSON that another agent or automation layer can consume.

```text
Screenshot
   ↓
Scale to bounded model budget
   ↓
Add ruler/grid overlay
   ↓
Vision model extracts controls
   ↓
Parse JSON and validate bounds
   ↓
Map scaled coordinates back to original screen coordinates
   ↓
Optional overlay verification
   ↓
Automation layer, after policy approval
```

## Key design choices

### 1. Pixel-space ruler instead of normalized coordinates

Earlier experiments used normalized 0–1000 coordinates. The current preferred version uses pixel-space ruler markings on the scaled image. That makes the prompt simpler and helps visual inspection: the image the model sees is the coordinate system.

### 2. Scale before analysis

The screenshot is scaled so its longest side is at most 768 pixels by default. This keeps image-token cost predictable and makes model behavior more stable. The output preserves the scale factor so boxes can be mapped back to the original screenshot.

### 3. Verification-first

The output should be verified before being trusted. The `overlay` command draws the detected boxes and centers back onto the original screenshot. This is intentionally aligned with a deterministic-engineering posture: use the LLM to extract, then verify with hard logic and human-visible artifacts.

## Output contract

The model returns controls in scaled-image coordinates:

```json
{
  "type": "button",
  "label": "Continue",
  "x": 580,
  "y": 450,
  "width": 150,
  "height": 25,
  "confidence": 0.82
}
```

The CLI augments each control with:

- `center`: scaled-image center point,
- `original`: original-screenshot x/y/width/height/center.

## Agent loop integration

A safe loop should separate observation from action:

```text
Vision AI: "I see a Create Project link centered at (1203, 202)."
Planner AI: "That matches the goal. Proposed action: click it."
Policy layer: "This is non-destructive and target confidence is high. Approved."
Executor: click(1203, 202)
Vision AI: observe again
```

## Privacy note

Screenshots may contain private information. Do not publish demo images containing chats, emails, addresses, tokens, API keys, or personal contacts. For the public repository, use synthetic screenshots or heavily redacted examples.
