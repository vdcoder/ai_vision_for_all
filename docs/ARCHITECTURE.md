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

### 2. White canvas with margin

The scaled image is pasted onto a white canvas with a fixed margin (default 50 px) on each side. Ruler labels are drawn in this margin. This separates the UI content from the ruler annotations so the model can't confuse the two, and gives the model clear visual breathing room at the edges — preventing the common behaviour of clustering predictions near boundaries.

### 3. Single click pixel, not bounding boxes

Earlier versions asked the model for bounding boxes (x, y, width, height). Models tend to default to element centres regardless, and boxes require the caller to derive a click point anyway. The current prompt asks for a single best click pixel. This is simpler for the model, simpler for the caller, and easier to verify.

### 4. Scale before analysis

The screenshot is scaled so its longest side is at most 768 pixels by default. This keeps image-token cost predictable and makes model behavior more stable. The output preserves the scale factor so boxes can be mapped back to the original screenshot.

### 5. Iterative zoom refinement

One pass gives the model wide context but limited pixel resolution on small controls. `--passes N` automates a zoom loop: after each pass, the best-confidence control's `original` coordinates are used as the centre of a tighter crop (`--crop-factor F`, default 0.5). Each axis is independently halved, so the model sees a zoomed-in ruler view of the exact region it identified. This dramatically improves accuracy on small or dense UIs without increasing the token budget per call.

All returned `original` coordinates are always in the **full original image** space regardless of how many crops were applied, so the caller sees a consistent coordinate system across passes.

### 6. Verification-first

The output should be verified before being trusted. The `overlay` command draws the detected boxes and centers back onto the original screenshot. This is intentionally aligned with a deterministic-engineering posture: use the LLM to extract, then verify with hard logic and human-visible artifacts.

## Output contract

The model returns a single best click pixel per control (canvas coordinate space):

```json
{
  "type": "button",
  "label": "Wi-Fi",
  "x": 358,
  "y": 165,
  "reason": "Centre of the circular Wi-Fi icon measured from ruler ticks.",
  "confidence": 0.97
}
```

The CLI validates bounds and augments each control with two additional coordinate representations:

- `scaled`: click point in the **scaled image** (canvas coordinates minus margin).
- `original`: click point in the **original full-resolution screenshot** — the value to use for automation.

No bounding boxes or rectangles are used. A single pixel is the click target — precise, unambiguous, and easy to verify with an overlay.

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
