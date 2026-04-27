# Basic Usage

## Install

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -e .
cp .env.example .env
# add your GEMINI_API_KEY to .env
```

---

## The three commands

### preview — no API call, inspect the ruler image

```bash
aivision preview screenshot.png --output outputs/ruler.png
```

### analyze — get structured click coordinates

```bash
# Single pass (fastest, good for obvious targets)
aivision analyze screenshot.png \
  --goal "launch Ubuntu 24.04 LTS" \
  --target "the Launch button for Ubuntu 24.04 Noble Numbat" \
  --output outputs/result.json

# Two passes (recommended for most targets)
aivision analyze screenshot.png \
  --goal "turn off Wi-Fi" \
  --target "the Wi-Fi toggle button" \
  --passes 2 \
  --preview outputs/ruler.png \
  --output outputs/result.json

# Four passes with a cheaper model (matches 2-pass Flash accuracy at ~1/4 the cost)
aivision analyze screenshot.png \
  --goal "turn off Wi-Fi" \
  --target "the Wi-Fi toggle button" \
  --model gemini-2.5-flash-lite \
  --passes 4 \
  --preview outputs/ruler.png \
  --output outputs/result.json
```

Multi-pass output files (with `--passes 2`, `--output outputs/result.json`):
```
outputs/result.pass1.json    first pass (full screenshot)
outputs/result.pass2.json    second pass (zoomed in)
outputs/result.json          copy of the final pass

outputs/ruler.pass1.png      what the model saw in pass 1
outputs/ruler.pass2.png      what the model saw in pass 2
outputs/ruler.png            copy of the final pass ruler
```

### overlay — draw the predicted click point back on the original

```bash
aivision overlay screenshot.png outputs/result.json \
  --output outputs/verify.png
```

---

## Reproducing the README demo images

### macOS — turn off Wi-Fi (2-pass gemini-2.5-flash)

```bash
aivision analyze ./inputs/mac_gui.png \
  --goal "turn off wifi" \
  --target "the Wi-Fi blue circular toggle button" \
  --model gemini-2.5-flash \
  --passes 2 \
  --preview outputs/mac_wifi_ruler.png \
  --output outputs/mac_wifi.json

aivision overlay ./inputs/mac_gui.png outputs/mac_wifi.json \
  --output outputs/mac_wifi_overlay.png
```

### Ubuntu — launch a VM (4-pass gemini-2.5-flash-lite)

```bash
aivision analyze ./inputs/ubuntu_gui.png \
  --goal "spin up an Ubuntu 24.04 virtual machine" \
  --target "the Launch button for Ubuntu 24.04 LTS Noble Numbat" \
  --model gemini-2.5-flash-lite \
  --passes 4 \
  --preview outputs/ubuntu_launch_ruler.png \
  --output outputs/ubuntu_launch.json

aivision overlay ./inputs/ubuntu_gui.png outputs/ubuntu_launch.json \
  --output outputs/ubuntu_launch_overlay.png
```

### Windows 11 — open a video file (2-pass gemini-2.5-flash)

```bash
aivision analyze ./inputs/win11_gui.png \
  --goal "open a video file to play" \
  --target "the VID 1.mp4 file row" \
  --model gemini-2.5-flash \
  --passes 2 \
  --preview outputs/win11_vid_ruler.png \
  --output outputs/win11_vid.json

aivision overlay ./inputs/win11_gui.png outputs/win11_vid.json \
  --output outputs/win11_vid_overlay.png
```
