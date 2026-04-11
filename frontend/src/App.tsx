import { useCallback, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type AnalyzeResponse = {
  centering: {
    left_right: string;
    top_bottom: string;
    lr_small_pct?: number;
    tb_small_pct?: number;
    margins_px?: { left: number; right: number; top: number; bottom: number };
  };
  estimated_grades: { PSA: number; BGS: number; CGC: number };
  warp_width?: number | null;
  warp_height?: number | null;
  detection_confidence: string;
  centering_method?: string | null;
  centering_build?: string | null;
};

export default function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onFile = useCallback(async (file: File | null) => {
    setError(null);
    setResult(null);
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_BASE}/analyze-card`, {
        method: "POST",
        body,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      const json = (await res.json()) as AnalyzeResponse;
      setResult(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="page">
      <header className="header">
        <h1>Card Grading AI</h1>
        <p className="tagline">
          Upload a trading card photo: centering metrics and rough grade hints (centering only).
        </p>
      </header>

      <section className="panel">
        <label className="upload">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
          />
          <span>Choose image</span>
        </label>
        {loading && <p className="status">Analyzing…</p>}
        {error && <p className="error">{error}</p>}
        <p className="api-hint">
          Backend: <span className="mono">{API_BASE}</span>
        </p>
      </section>

      <div className="layout">
        {previewUrl && (
          <figure className="preview">
            <img src={previewUrl} alt="Selected card" />
          </figure>
        )}
        {result && (
          <section className="results panel">
            <h2>Results</h2>
            <dl>
              <dt>Left / right</dt>
              <dd>{result.centering.left_right}</dd>
              <dt>Top / bottom</dt>
              <dd>{result.centering.top_bottom}</dd>
              <dt>PSA (approx.)</dt>
              <dd>{result.estimated_grades.PSA}</dd>
              <dt>BGS (approx.)</dt>
              <dd>{result.estimated_grades.BGS}</dd>
              <dt>CGC (approx.)</dt>
              <dd>{result.estimated_grades.CGC}</dd>
              <dt>Detection</dt>
              <dd>{result.detection_confidence}</dd>
              {result.centering_method != null && result.centering_method !== "" && (
                <>
                  <dt>Centering method</dt>
                  <dd className="mono">{result.centering_method}</dd>
                </>
              )}
              {result.centering_build != null && result.centering_build !== "" && (
                <>
                  <dt>Pipeline build</dt>
                  <dd className="mono">{result.centering_build}</dd>
                </>
              )}
              {result.warp_width != null && result.warp_height != null && (
                <>
                  <dt>Normalized crop</dt>
                  <dd>
                    {result.warp_width} × {result.warp_height} px
                  </dd>
                </>
              )}
            </dl>
          </section>
        )}
      </div>
    </div>
  );
}
