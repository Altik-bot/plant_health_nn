# Leaf Disease Detection Demo — Clinical Instrument

A single page at `/` that lets you upload a leaf image, send it to your model API, and read back the predicted class, confidence, ranked alternates, and the Grad-CAM heatmap.

## API wiring

Confirmed from your live API: it exposes exactly one endpoint, `POST /predict`, accepting `multipart/form-data` with a single `file` field.

- Base URL used exactly as given: `https://triage-kiln-savage.ngrok-free.dev/docs`
- Predict call: `POST /predict` with `FormData { file }`
- The header link "API Documentation" points to `/docs`
- Requests include `ngrok-skip-browser-warning: true` so the free ngrok tunnel returns JSON instead of its interstitial page
- The response is rendered defensively: the UI reads common field names (`class`/`prediction`/`label`, `confidence`/`probability`, `probabilities`/`top_k`, `gradcam`/`heatmap` as base64 or URL) and, if a field is absent, hides that panel and shows the raw JSON in a collapsible block so you can see exactly what came back
- Grad-CAM renders whether the API returns a data URL, a raw base64 string, or an absolute image URL

## Page structure (matching the chosen direction)

- Top bar: KLN mark, "Triage Kiln", subtitle, right-side links (Interface, API Documentation) plus a live API status dot
- Left column: Input Module — dashed-border square drop/click zone showing the exact uploaded image, hover "Replace Image", two metadata tiles (File Size, Format/dimensions), and a full-width "Initialize Prediction" button with a scanning-line loading state
- Right column: Classification Engine card (accent left border, big class name, mono confidence percentage, short description line), then a two-up row of Ranked Alternates (mono list) and Grad-CAM Visualizer with the "Attention Mask" caption
- Below: 3-column Processing Pipeline explainer (preprocessing / feature extraction / softmax inference)
- Footer: mono meta line with measured request latency
- Empty and error states: before any run, results panels show a quiet "awaiting input" state; failed requests show a clear error strip with the reason (network, CORS, non-200)

## Design tokens

Ported verbatim from the chosen direction: near-white `#fdfdfd` background, ink `#0f172a`, muted `#64748b`, emerald accent `#10b981`, border `#e2e8f0`, sharp corners, Inter for text and JetBrains Mono for all numbers/labels, plus the `scan` and `fade-in` animations.

## CORS on your FastAPI side

The browser calls your API directly, so your FastAPI app must allow it. I'll include this snippet in the page (a small "Backend setup" note near the API link) so it's copy-pasteable:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

I cannot edit your Python service from here — you'll need to add that and restart it. Until then, predictions will fail with a CORS error, which the page reports explicitly with this fix inline.

## Technical notes

- Frontend only, no backend or database added
- Route: rewrite `src/routes/index.tsx`, with its own SEO head (title, description, og/twitter)
- Fonts loaded via `<link>` in `src/routes/__root.tsx`; tokens added to `src/styles.css`
- One generated hero-free asset set: a sample leaf image and a sample heatmap image used only as the pre-run placeholder visuals
