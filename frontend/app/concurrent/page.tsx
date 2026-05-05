import NavBar from "@/components/NavBar";
import ConcurrentRunner from "@/components/ConcurrentRunner";
import VramTimeSeries from "@/components/VramTimeSeries";

export const metadata = {
  title: "Concurrent Cases — AOB",
  description: "Run 3 cancer cases simultaneously to demonstrate MI300X's multi-model capacity",
};

export default function ConcurrentPage() {
  return (
    <>
      <NavBar />
      <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "3rem 2rem 5rem" }}>

        {/* Header */}
        <div style={{ marginBottom: "2.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.6rem" }}>
            <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--teal-light)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Production Readiness
            </span>
          </div>
          <h1 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "0.6rem", lineHeight: 1.2 }}>
            Concurrent Case Processing
          </h1>
          <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: "700px" }}>
            A real deployment serves multiple patients simultaneously. Launch 3 different cancer cases at once and watch all
            three pipelines run in parallel — pathologists, researchers, and oncologists for each, all sharing the MI300X&apos;s
            192 GB memory pool without model swapping.
          </p>
        </div>

        {/* Concurrent runner */}
        <ConcurrentRunner />

        {/* VRAM time series */}
        <div style={{ marginTop: "2rem" }}>
          <h2
            style={{
              fontSize: "1.1rem",
              fontWeight: 700,
              color: "var(--text-primary)",
              marginBottom: "1rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            📈 Live VRAM — 5-Minute Window
          </h2>
          <VramTimeSeries pollIntervalMs={2000} windowSeconds={300} showH100Line />
        </div>

        {/* Architecture note */}
        <div
          style={{
            marginTop: "2rem",
            padding: "1rem 1.25rem",
            background: "rgba(13,148,136,0.06)",
            border: "1px solid var(--teal-border)",
            borderRadius: "10px",
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            lineHeight: 1.7,
          }}
        >
          <strong style={{ color: "var(--teal-light)" }}>How this works:</strong> Each concurrent case gets its own
          independent board run. GigaPath is shared (loaded once, batched across cases). Llama 3.3 70B serves all three
          cases through vLLM&apos;s continuous batching. The KV cache grows proportionally — watch the VRAM bar climb above the
          H100&apos;s 80 GB OOM threshold.
        </div>

      </main>
    </>
  );
}
