import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";

const API_DOCS_URL = "https://triage-kiln-savage.ngrok-free.dev/docs";
const API_BASE = API_DOCS_URL.replace(/\/docs\/?$/, "");
const PREDICT_URL = `${API_BASE}/predict`;

type PredictionResult = {
  class: string;
  confidence: number;
  heatmap: string | null;
};

function hexToBase64(hex: string): string | null {
  if (hex.length % 2 !== 0) return null;
  try {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
    }
    let binary = "";
    const chunk = 8192;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  } catch {
    return null;
  }
}

function toImageSrc(raw: string): string | null {
  const value = raw.trim();
  if (!value) return null;
  if (value.startsWith("data:")) return value;

  // Some backends return the JPEG bytes hex-encoded rather than base64.
  if (/^[0-9a-fA-F]+$/.test(value) && value.length % 2 === 0) {
    const b64 = hexToBase64(value);
    if (b64) return `data:image/jpeg;base64,${b64}`;
  }

  return `data:image/jpeg;base64,${value}`;
}

function parseResult(json: unknown): PredictionResult {
  const obj = (json && typeof json === "object" ? json : {}) as Record<string, unknown>;

  const cls =
    typeof obj["class"] === "string" && obj["class"].trim()
      ? obj["class"].trim()
      : "Unknown";

  const confidence =
    typeof obj["confidence"] === "number" && Number.isFinite(obj["confidence"])
      ? obj["confidence"]
      : 0;

  const heatmapValue =
    typeof obj["heatmap"] === "string" && obj["heatmap"].trim()
      ? obj["heatmap"].trim()
      : null;

  const heatmap = heatmapValue ? toImageSrc(heatmapValue) : null;

  return { class: cls, confidence, heatmap };
}

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Plant_health_nn — Plant Leaf Disease Detection" },
      {
        name: "description",
        content:
          "Upload a leaf image and get a neural network disease prediction with confidence score and Grad-CAM heatmap.",
      },
      { property: "og:title", content: "Plant_health_nn — Plant Leaf Disease Detection" },
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  const confidencePercent = useMemo(() => {
    if (!result) return null;
    const value = result.confidence <= 1 ? result.confidence * 100 : result.confidence;
    return value.toFixed(1);
  }, [result]);

  function onSelect(next: File | null) {
    if (!next) return;
    setFile(next);
    setResult(null);
    setError(null);
    setLatency(null);

    const url = URL.createObjectURL(next);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
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
        setResult(parseResult(JSON.parse(text)));
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

  return (
    <div className="min-h-screen bg-background p-6 font-sans text-foreground md:p-12">
      <header className="mx-auto mb-12 flex max-w-6xl flex-col gap-4 border-b border-border pb-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-lg font-bold tracking-tight">Plant_health_nn</h1>
          <p className="font-mono text-xs text-muted">
            Plant leaf disease detection with Grad-CAM visualization
          </p>
        </div>
        <a
          href={API_DOCS_URL}
          target="_blank"
          rel="noreferrer"
          className="font-mono text-xs text-muted underline underline-offset-4 transition-colors hover:text-foreground"
        >
          API Documentation
        </a>
      </header>

      <main className="mx-auto max-w-6xl space-y-10">
        <section className="grid grid-cols-1 gap-8 lg:grid-cols-12">
          <div className="space-y-6 lg:col-span-5">
            <div>
              <h2 className="mb-3 text-xs font-bold uppercase tracking-widest text-muted">
                Upload Leaf Image
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
                      alt="Uploaded leaf specimen"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <span className="font-mono text-[10px] uppercase tracking-widest text-muted">
                      Drop or select image
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

            <button
              onClick={predict}
              disabled={!file || loading}
              className="flex w-full items-center justify-center gap-3 bg-foreground py-4 text-xs font-bold uppercase tracking-[0.2em] text-background transition-all hover:bg-foreground/90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Analyzing..." : "Run Prediction"}
              {loading && <span className="size-2 animate-pulse rounded-full bg-accent" />}
            </button>

            {error && (
              <div className="border-l-4 border-destructive bg-slate-50 p-4">
                <p className="font-mono text-[10px] uppercase tracking-widest text-destructive">
                  Error
                </p>
                <p className="mt-2 text-xs leading-relaxed text-muted">{error}</p>
              </div>
            )}
          </div>

          <div className="space-y-8 lg:col-span-7">
            {result && (
              <div className="animate-reveal border-l-4 border-accent bg-slate-50 p-6">
                <div className="mb-1 text-xs font-bold uppercase tracking-widest text-muted">
                  Prediction Result
                </div>
                <div className="flex flex-wrap items-baseline gap-4">
                  <span className="text-3xl font-bold tracking-tight">{result.class}</span>
                  <span className="font-mono text-2xl font-medium text-accent">
                    {confidencePercent}%
                  </span>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="animate-reveal [animation-delay:100ms]">
                <h3 className="mb-3 text-xs font-bold uppercase tracking-widest text-muted">
                  Original Input
                </h3>
                <div className="relative aspect-square bg-slate-100 ring-1 ring-black/5">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Original uploaded leaf image"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center font-mono text-[10px] uppercase tracking-widest text-muted">
                      No image selected
                    </div>
                  )}
                </div>
              </div>

              <div className="animate-reveal [animation-delay:200ms]">
                <h3 className="mb-3 text-xs font-bold uppercase tracking-widest text-muted">
                  Grad-CAM / Model Attention
                </h3>
                <div className="relative aspect-square bg-slate-100 ring-1 ring-black/5">
                  {result?.heatmap ? (
                    <img
                      src={result.heatmap}
                      alt="Grad-CAM heatmap highlighting regions the model focused on"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center font-mono text-[10px] uppercase tracking-widest text-muted">
                      {loading ? "Generating heatmap..." : "Awaiting prediction"}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="animate-reveal border-t border-border pt-10">
          <h2 className="mb-6 text-xs font-bold uppercase tracking-widest">Processing Pipeline</h2>
          <div className="grid grid-cols-1 gap-6 text-sm leading-relaxed md:grid-cols-3">
            <div className="space-y-2">
              <span className="font-mono text-accent">01 / Preprocessing</span>
              <p className="text-muted">
                Image is resized and normalized to match the ResNet18 training input.
              </p>
            </div>
            <div className="space-y-2">
              <span className="font-mono text-accent">02 / Feature Extraction</span>
              <p className="text-muted">
                ResNet18 convolutional layers extract texture, color, and shape patterns from the
                leaf.
              </p>
            </div>
            <div className="space-y-2">
              <span className="font-mono text-accent">03 / Grad-CAM Explanation</span>
              <p className="text-muted">
                Grad-CAM produces a heatmap showing which regions most influenced the final
                prediction.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="mx-auto mt-16 flex max-w-6xl items-center justify-between border-t border-border pt-6 font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        <span>{"\n"}</span>
        <div className="flex gap-6">
          <span>Latency: {latency !== null ? `${latency}ms` : "—"}</span>
          <span>Endpoint: /predict</span>
        </div>
      </footer>
    </div>
  );
}
