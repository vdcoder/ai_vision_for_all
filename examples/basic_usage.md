# Basic Usage

```bash
pip install -e .
cp .env.example .env
```

Add your Gemini API key to `.env`.

```bash
aivision preview screenshot.png -o outputs/ruler_preview.png

aivision analyze screenshot.png \
  --goal "open settings" \
  --preview outputs/ruler_preview.png \
  -o outputs/result.json

aivision overlay screenshot.png outputs/result.json \
  -o outputs/verify_overlay.png
```

Suggested demo flow for the README or LinkedIn:

1. Use a synthetic app screenshot or a public demo UI.
2. Show original screenshot.
3. Show ruler screenshot.
4. Show JSON result.
5. Show verification overlay.

Avoid private screenshots in the public repo.
