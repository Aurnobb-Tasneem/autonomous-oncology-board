"use client";

import { useEffect, useMemo, useState } from "react";
import { getHeatmaps } from "@/lib/api";

function SkeletonGrid() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "0.5rem" }}>
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="glass-card-sm"
          style={{
            borderRadius: "6px",
            aspectRatio: "1",
            background: "linear-gradient(110deg, rgba(148,163,184,0.06) 20%, rgba(148,163,184,0.12) 40%, rgba(148,163,184,0.06) 60%)",
            backgroundSize: "200% 100%",
            animation: "aob-skel 1.2s ease-in-out infinite",
            borderColor: "rgba(15,118,110,0.18)",
          }}
        />
      ))}
      <style>{`
        @keyframes aob-skel {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}

export default function HeatmapViewer({ jobId }: { jobId: string }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);
  const [heatmaps, setHeatmaps] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    let attempts = 0;
    let timer: number | null = null;

    const tick = async () => {
      attempts += 1;
      setError(null);
      try {
        const res = await getHeatmaps(jobId);
        if (cancelled) return;
        setPending(res.pending);
        if (res.heatmaps?.length) {
          setHeatmaps(res.heatmaps);
          setLoading(false);
          return;
        }
        // Still generating; keep a light polling loop for a short window.
        setLoading(true);
        const delay = attempts < 6 ? 1200 : 2000;
        timer = window.setTimeout(tick, delay);
      } catch {
        if (cancelled) return;
        setError("Failed to load heatmaps.");
        setLoading(false);
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId]);

  const headerText = useMemo(() => {
    if (loading) return pending ? "Generating heatmaps…" : "Loading heatmaps…";
    if (error) return "Heatmaps unavailable";
    if (!heatmaps.length) return "Heatmaps not yet available";
    return `${heatmaps.length} patch${heatmaps.length !== 1 ? "es" : ""} analysed`;
  }, [loading, pending, error, heatmaps.length]);

  if (loading && heatmaps.length === 0) {
    return (
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>{headerText}</span>
          <button className="btn-ghost" style={{ fontSize: "0.78rem", padding: "0.25rem 0.65rem" }} disabled>
            🔴 Hide AI Attention
          </button>
        </div>
        <SkeletonGrid />
        <div style={{ marginTop: "0.75rem", fontSize: "0.78rem", color: "var(--text-muted)", textAlign: "center" }}>
          {pending ? "GigaPath is still generating attention overlays. This usually takes a few seconds." : "Fetching attention overlays…"}
        </div>
      </div>
    );
  }

  if (error && heatmaps.length === 0) {
    return (
      <div
        style={{
          textAlign: "center",
          color: "var(--danger)",
          padding: "1.5rem",
          fontSize: "0.85rem",
          border: "1px dashed var(--border)",
          borderRadius: "8px",
        }}
      >
        {error}
      </div>
    );
  }

  if (!heatmaps || heatmaps.length === 0) {
    return (
      <div
        style={{
          textAlign: "center",
          color: "var(--text-muted)",
          padding: "1.5rem",
          fontSize: "0.85rem",
          border: "1px dashed var(--border)",
          borderRadius: "8px",
        }}
      >
        Heatmaps not yet available
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>{headerText}</span>
        <button
          className="btn-ghost"
          style={{ fontSize: "0.78rem", padding: "0.25rem 0.65rem" }}
          onClick={() => setShowOverlay((v) => !v)}
        >
          {showOverlay ? "🔴 Hide AI Attention" : "🟢 Show AI Attention"}
        </button>
      </div>

      {/* Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
          gap: "0.5rem",
        }}
      >
        {heatmaps.map((b64, i) => (
          <div
            key={i}
            onClick={() => setSelected(selected === i ? null : i)}
            style={{
              position: "relative",
              cursor: "pointer",
              borderRadius: "6px",
              overflow: "hidden",
              border: selected === i ? "2px solid var(--teal-light)" : "1px solid var(--border-teal)",
              transition: "border 0.2s ease",
              aspectRatio: "1",
              background: "#0d1b30",
            }}
          >
            <img
              src={`data:image/png;base64,${b64}`}
              alt={`Patch ${i + 1}`}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                opacity: showOverlay ? 1 : 0.5,
                transition: "opacity 0.3s ease",
              }}
            />
            <div
              style={{
                position: "absolute",
                bottom: 0,
                left: 0,
                right: 0,
                background: "rgba(0,0,0,0.6)",
                fontSize: "0.68rem",
                color: "var(--text-muted)",
                textAlign: "center",
                padding: "0.15rem 0",
              }}
            >
              Patch {i + 1}
            </div>
          </div>
        ))}
      </div>

      {/* Expanded view */}
      {selected !== null && (
        <div
          style={{
            marginTop: "0.75rem",
            borderRadius: "8px",
            overflow: "hidden",
            border: "1px solid var(--teal-border)",
            background: "#0d1b30",
          }}
        >
          <img
            src={`data:image/png;base64,${heatmaps[selected]}`}
            alt={`Patch ${selected + 1} enlarged`}
            style={{ width: "100%", display: "block" }}
          />
          <div style={{ padding: "0.5rem", fontSize: "0.78rem", color: "var(--text-muted)", textAlign: "center" }}>
            Patch {selected + 1} — GigaPath Attention Rollout · Red = Suspicious Region
          </div>
        </div>
      )}
    </div>
  );
}
