import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";

import sampleGradcam from "../assets/sample-gradcam.jpg";

const API_DOCS_URL = "https://triage-kiln-savage.ngrok-free.dev/docs";
const API_BASE = API_DOCS_URL.replace(/\/docs\/?$/, "");
const PREDICT_URL = `${API_BASE}/predict`;

const CORS_SNIPPET = `from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)`;

type Alternate = { label: string; value: number };

type Parsed = {
  label: string | null;
  confidence: number | null;
  alternates: Alternate[];
  gradcam: string | null;
  raw: unknown;
};

function asPercent(value: number) {
  const pct = value <= 1 ? value * 100 : value;
  return pct;
}

function pickString(obj: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const v = obj[key];
    if (typeof v === "string" && v.trim()) return v;
  }
  return null;
}

function pickNumber(obj: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const v = obj[key];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() && Number.isFinite(Number(v))) return Number(v);
  }
  return null;
}

function toImageSrc(value: string) {
  if (value.startsWith("data:") || /^https?:\/\//.test(value)) return value;
  if (value.startsWith("/")) return `${API_BASE}${value}`;
  return `data:image/png;base64,${value}`;
}

function parseResponse(json: unknown): Parsed {
  const obj = (json && typeof json === "object" ? json : {}) as Record<string, unknown>;

  const label =
    pickString(obj, ["class", "prediction", "predicted_class", "label", "disease", "name"]) ??
    null;
  const rawConfidence = pickNumber(obj, ["confidence", "probability", "score", "prob"]);

  const alternates: Alternate[] = [];
  const probSource = (obj["probabilities"] ?? obj["top_k"] ?? obj["predictions"]) as unknown;

  if (Array.isArray(probSource)) {
    for (const item of probSource) {
      if (item && typeof item === "object") {
        const row = item as Record<string, unknown>;
        const l = pickString(row, ["class", "label", "name", "prediction"]);
        const v = pickNumber(row, ["confidence", "probability", "score", "value", "prob"]);
        if (l && v !== null) alternates.push({ label: l, value: asPercent(v) });
      }
    }
  } else if (probSource && typeof probSource === "object") {
    for (const [l, v] of Object.entries(probSource as Record<string, unknown>)) {
      if (typeof v === "number") alternates.push({ label: l, value: asPercent(v) });
    }
  }

  alternates.sort((a, b) => b.value - a.value);

  const gradcamRaw = pickString(obj, [
    "gradcam",
    "grad_cam",
    "gradcam_image",
    "heatmap",
    "heatmap_image",
    "cam",
    "overlay",
  ]);

  return {
    label,
    confidence: rawConfidence === null ? null : asPercent(rawConfidence),
    alternates,
    gradcam: gradcamRaw ? toImageSrc(gradcamRaw) : null,
    raw: json,
  };
}

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Triage Kiln — Plant Leaf Disease Detection Demo" },
      {
        name: "description",
        content:
          "Upload a leaf image and get a neural network disease prediction with confidence score and Grad-CAM heatmap explaining the model's decision.",
      },
      { property: "og:title", content: "Triage Kiln — Plant Leaf Disease Detection Demo" },
      {
        property: "og:description",
        content:
          "Neural diagnostic terminal for plant pathology: upload a leaf, read the predicted class, confidence and Grad-CAM attention map.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Parsed | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  const fileSize = useMemo(() => {
    if (!file) return "—";
    const mb = file.size / (1024 * 1024);
    return mb >= 1 ? `${mb.toFixed(2)} MB` : `${Math.round(file.size / 1024)} KB`;
  }, [file]);

  const format = useMemo(() => {
    if (!file) return "—";
    const ext = (file.type.split("/")[1] ?? "img").toUpperCase();
    return dims ? `${ext} / ${dims.w}×${dims.h}` : ext;
  }, [file, dims]);

  function onSelect(next: File | null) {
    if (!next) return;
    setFile(next);
    setResult(null);
    setError(null);
    setLatency(null);
    setDims(null);
    const url = URL.createObjectURL(next);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
    const img = new Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = url;
  }

  async function predict() {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const started = performance.now();
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(PREDICT_URL, {
        method: "POST",
        body,
        headers: { "ngrok-skip-browser-warning": "true" },
      });
      setLatency(Math.round(performance.now() - started));
      const text = await res.text();
      if (!res.ok) {
        setError(`API returned ${res.status}. ${text.slice(0, 240)}`);
        return;
      }
      try {
        setResult(parseResponse(JSON.parse(text)));
      } catch {
        setError(`Response was not JSON. ${text.slice(0, 240)}`);
      }
    } catch (e) {
      setLatency(null);
      setError(
        `Request failed — most likely a CORS block or the tunnel being offline. ${
          e instanceof Error ? e.message : String(e)
        }`,
      );
    } finally {
      setLoading(false);
    }
  }

  const alternates = result?.alternates.filter((a) => a.label !== result.label).slice(0, 4) ?? [];

  return (
    <div className="min-h-screen bg-background p-6 font-sans text-foreground md:p-12">
      <nav className="mx-auto mb-16 flex max-w-7xl items-center justify-between border-b border-border pb-6">
        <div className="flex items-center gap-4">
          <div className="flex size-8 items-center justify-center bg-foreground font-mono text-xs font-bold text-background">
            KLN
          </div>
          <div>
            <h1 className="text-xs font-bold uppercase tracking-widest">Triage Kiln v2.0</h1>
            <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
              Neural Diagnostic Terminal / Plant Pathology
            </p>
          </div>
        </div>
        <div className="hidden gap-8 font-mono text-[10px] uppercase tracking-widest text-muted md:flex">
          <span className="text-foreground underline underline-offset-4">Interface</span>
          <a
            href={API_DOCS_URL}
            target="_blank"
            rel="noreferrer"
            className="transition-colors hover:text-foreground"
          >
            API Documentation
          </a>
          <span className="text-accent">● System Ready</span>
        </div>
      </nav>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-12 lg:grid-cols-12">
        <section className="animate-reveal space-y-8 lg:col-span-5">
          <div>
            <h2 className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
              <span className="size-1.5 rounded-full bg-foreground" />
              Input Module
            </h2>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                onSelect(e.dataTransfer.files?.[0] ?? null);
              }}
              onClick={() => inputRef.current?.click()}
              className="group relative flex aspect-square cursor-pointer flex-col items-center justify-center border border-dashed border-border bg-slate-50/50 p-4 transition-colors hover:bg-slate-50"
            >
              <div className="relative flex h-full w-full items-center justify-center overflow-hidden bg-white outline-1 -outline-offset-1 outline-black/5">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Uploaded leaf specimen sent to the model"
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted">
                    Drop_Or_Select_Image
                  </span>
                )}
                {loading && (
                  <div className="animate-scan absolute inset-0 z-10 border-b border-accent/30 bg-accent/5" />
                )}
              </div>
              <div className="absolute inset-0 flex items-center justify-center bg-background/80 opacity-0 transition-opacity group-hover:opacity-100">
                <span className="bg-foreground px-6 py-3 text-[10px] font-bold uppercase tracking-widest text-background">
                  {previewUrl ? "Replace Image" : "Select Image"}
                </span>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="border border-border p-4">
              <span className="mb-1 block font-mono text-[9px] uppercase text-muted">
                File Size
              </span>
              <span className="font-mono text-sm">{fileSize}</span>
            </div>
            <div className="border border-border p-4">
              <span className="mb-1 block font-mono text-[9px] uppercase text-muted">Format</span>
              <span className="font-mono text-sm">{format}</span>
            </div>
          </div>

          <button
            onClick={predict}
            disabled={!file || loading}
            className="flex w-full items-center justify-center gap-3 bg-foreground py-4 text-xs font-bold uppercase tracking-[0.2em] text-background transition-all hover:bg-foreground/90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Running Inference" : "Initialize Prediction"}
            <span className="size-2 animate-pulse rounded-full bg-accent" />
          </button>

          {error && (
            <div className="border-l-4 border-destructive bg-slate-50 p-4">
              <p className="font-mono text-[10px] uppercase tracking-widest text-destructive">
                Request Error
              </p>
              <p className="mt-2 text-[11px] leading-relaxed text-muted">{error}</p>
            </div>
          )}

          <details className="border border-border p-4">
            <summary className="cursor-pointer font-mono text-[9px] uppercase tracking-widest text-muted">
              Backend Setup / Enable CORS in FastAPI
            </summary>
            <pre className="mt-4 overflow-x-auto whitespace-pre font-mono text-[10px] leading-relaxed text-foreground">
              {CORS_SNIPPET}
            </pre>
          </details>
        </section>

        <section className="space-y-10 lg:col-span-7">
          <div className="animate-reveal [animation-delay:150ms]">
            <h2 className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
              <span className="size-1.5 rounded-full bg-foreground" />
              Classification Engine
            </h2>
            <div className="border-l-4 border-accent bg-slate-50 p-8">
              <div className="mb-2 flex items-start justify-between gap-6">
                <h3 className="text-3xl font-bold tracking-tighter">
                  {result?.label ?? "Awaiting input"}
                </h3>
                <span className="font-mono text-2xl font-medium text-accent">
                  {result?.confidence !== null && result?.confidence !== undefined
                    ? `${result.confidence.toFixed(1)}%`
                    : "—"}
                </span>
              </div>
              <p className="max-w-md text-sm leading-relaxed text-muted">
                {result
                  ? "Prediction returned by the model API. Confidence is the softmax probability of the top class across the trained disease categories."
                  : "Upload a leaf image and run the prediction. The detected disease class and its confidence score appear here."}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            <div className="animate-reveal [animation-delay:300ms]">
              <h4 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-muted">
                Ranked Alternates
              </h4>
              <div className="space-y-3 font-mono text-[11px]">
                {alternates.length > 0 ? (
                  alternates.map((alt, i) => (
                    <div
                      key={alt.label}
                      className="flex items-center justify-between border-b border-border pb-2"
                    >
                      <span className="text-foreground">
                        {String(i + 2).padStart(2, "0")} / {alt.label}
                      </span>
                      <span className="text-muted">{alt.value.toFixed(1)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="text-muted">
                    {result ? "No ranked probabilities in response." : "Awaiting prediction."}
                  </p>

                )}
              </div>
            </div>

            <div className="animate-reveal [animation-delay:450ms]">
              <h4 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-muted">
                Grad-CAM Visualizer
              </h4>
              <div className="relative aspect-video bg-slate-100 ring-1 ring-black/5">
                {result?.gradcam ? (
                  <img
                    src={result.gradcam}
                    alt="Grad-CAM heatmap showing the leaf regions that influenced the prediction"
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <>
                    <img
                      src={sampleGradcam}
                      alt="Example Grad-CAM activation map"
                      width={800}
                      height={512}
                      loading="lazy"
                      className="h-full w-full object-cover opacity-25"
                    />
                    <span className="absolute inset-0 grid place-items-center font-mono text-[9px] uppercase tracking-widest text-muted">
                      Activation_Map.png
                    </span>
                  </>
                )}
                <div className="absolute bottom-2 left-2 bg-foreground px-2 py-1 font-mono text-[8px] uppercase tracking-tighter text-background">
                  Attention Mask: Conv_Layer_17
                </div>
              </div>
            </div>
          </div>

          {result && (
            <div className="animate-reveal">
              <button
                onClick={() => setShowRaw((v) => !v)}
                className="font-mono text-[9px] uppercase tracking-widest text-muted underline underline-offset-4 hover:text-foreground"
              >
                {showRaw ? "Hide raw response" : "Show raw response"}
              </button>
              {showRaw && (
                <pre className="mt-4 max-h-64 overflow-auto border border-border p-4 font-mono text-[10px] leading-relaxed text-muted">
                  {JSON.stringify(result.raw, null, 2)}
                </pre>
              )}
            </div>
          )}

          <div className="animate-reveal border-t border-border pt-10 [animation-delay:600ms]">
            <h2 className="mb-6 text-xs font-bold uppercase tracking-widest">
              Processing Pipeline
            </h2>
            <div className="grid grid-cols-1 gap-6 text-[11px] leading-relaxed md:grid-cols-3">
              <div className="space-y-2">
                <span className="font-mono text-accent">01/Preprocessing</span>
                <p className="text-muted">
                  Image resized and normalized to match training conditions before it reaches the
                  network.
                </p>
              </div>
              <div className="space-y-2">
                <span className="font-mono text-accent">02/Feature Extraction</span>
                <p className="text-muted">
                  A pretrained convolutional network analyzes texture, color patterns, and shapes
                  across the leaf surface.
                </p>
              </div>
              <div className="space-y-2">
                <span className="font-mono text-accent">03/Softmax Inference</span>
                <p className="text-muted">
                  The final layer produces class probabilities; the highest becomes the prediction,
                  and Grad-CAM exposes the regions behind it.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="mx-auto mt-20 flex max-w-7xl items-center justify-between border-t border-border pt-6 font-mono text-[9px] uppercase tracking-[0.2em] text-muted">
        <span>Kiln Systems / Demo Environment</span>
        <div className="flex gap-6">
          <span>Latency: {latency !== null ? `${latency}ms` : "—"}</span>
          <span>Endpoint: /predict</span>
        </div>
      </footer>
    </div>
  );
}
