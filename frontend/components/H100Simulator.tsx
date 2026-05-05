"use client";

import { useState } from "react";

const H100_LIMIT_GB = 80;
const MI300X_TOTAL_GB = 192;

interface Component {
  id: string;
  label: string;
  gb: number;
  color: string;
  required: boolean;
  description: string;
}

const COMPONENTS: Component[] = [
  { id: "gigapath", label: "GigaPath ViT-Giant", gb: 3, color: "#0d9488", required: true, description: "Vision Foundation Model (1.1B params, FP16)" },
  { id: "qwen_vl", label: "Qwen2.5-VL 7B", gb: 15, color: "#22c55e", required: false, description: "Second-opinion VLM (BF16)" },
  { id: "llama_70b", label: "Llama 3.3 70B", gb: 70, color: "#0891b2", required: true, description: "Lead Oncologist LLM (FP8 quantized)" },
  { id: "lora_suite", label: "LoRA Suite ×3", gb: 16, color: "#38bdf8", required: false, description: "TNM + Biomarker + Treatment adapters (rank-16)" },
  { id: "kv_cache", label: "KV Cache (32k ctx)", gb: 30, color: "#7c3aed", required: false, description: "Transformer attention key-value cache" },
  { id: "qdrant", label: "Qdrant + Overhead", gb: 9, color: "#64748b", required: true, description: "Vector DB + ROCm runtime overhead" },
];

function Bar({
  used,
  total,
  limit,
  label,
  oomLabel,
  segments,
}: {
  used: number;
  total: number;
  limit: number;
  label: string;
  oomLabel?: string;
  segments: { label: string; gb: number; color: string }[];
}) {
  const pct = Math.min(100, (used / total) * 100);
  const overLimit = used > limit;

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "0.75rem",
          marginBottom: "0.4rem",
        }}
      >
        <span style={{ fontWeight: 700, color: overLimit ? "#f87171" : "var(--text-primary)" }}>
          {label}
        </span>
        <span
          style={{
            fontSize: "0.7rem",
            fontWeight: 700,
            color: overLimit ? "#f87171" : "#4ade80",
            background: overLimit ? "rgba(239,68,68,0.12)" : "rgba(74,222,128,0.1)",
            border: `1px solid ${overLimit ? "rgba(239,68,68,0.4)" : "rgba(74,222,128,0.3)"}`,
            padding: "0.15rem 0.5rem",
            borderRadius: "20px",
          }}
        >
          {overLimit ? `💥 OOM +${(used - limit).toFixed(0)} GB` : `${used.toFixed(0)} / ${total} GB`}
        </span>
      </div>

      {/* Segmented fill bar */}
      <div
        style={{
          height: "20px",
          background: "rgba(255,255,255,0.05)",
          borderRadius: "6px",
          overflow: "hidden",
          display: "flex",
          position: "relative",
        }}
      >
        {segments.map((seg, i) => {
          const segPct = Math.min((seg.gb / total) * 100, 100 - segments.slice(0, i).reduce((s, s2) => s + (s2.gb / total) * 100, 0));
          return (
            <div
              key={seg.label}
              title={`${seg.label}: ${seg.gb} GB`}
              style={{
                height: "100%",
                width: `${segPct}%`,
                background: seg.color,
                opacity: 0.85,
                transition: "width 0.5s ease",
                flexShrink: 0,
              }}
            />
          );
        })}

        {/* OOM overlay if over limit */}
        {overLimit && (
          <div
            style={{
              position: "absolute",
              left: `${(limit / total) * 100}%`,
              top: 0,
              bottom: 0,
              right: 0,
              background: "repeating-linear-gradient(45deg, rgba(239,68,68,0.3), rgba(239,68,68,0.3) 4px, transparent 4px, transparent 8px)",
              borderLeft: "2px solid #ef4444",
            }}
          />
        )}
      </div>

      {/* Limit marker */}
      <div style={{ position: "relative", height: "14px", marginTop: "2px" }}>
        <div
          style={{
            position: "absolute",
            left: `${(limit / total) * 100}%`,
            transform: "translateX(-50%)",
            fontSize: "0.6rem",
            color: overLimit ? "#f87171" : "#64748b",
            whiteSpace: "nowrap",
          }}
        >
          {oomLabel ?? `${limit} GB`}
        </div>
      </div>
    </div>
  );
}

export default function H100Simulator() {
  const [enabled, setEnabled] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(COMPONENTS.map((c) => [c.id, true]))
  );

  const toggle = (id: string) => {
    const comp = COMPONENTS.find((c) => c.id === id);
    if (comp?.required) return; // required components can't be toggled
    setEnabled((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const activeComponents = COMPONENTS.filter((c) => enabled[c.id]);
  const totalGb = activeComponents.reduce((sum, c) => sum + c.gb, 0);

  const mi300xSegments = activeComponents.map((c) => ({ label: c.label, gb: c.gb, color: c.color }));
  const h100Segments = activeComponents.map((c) => ({
    label: c.label,
    gb: Math.min(c.gb, Math.max(0, H100_LIMIT_GB - activeComponents.slice(0, activeComponents.indexOf(c)).reduce((s, s2) => s + s2.gb, 0))),
    color: c.color,
  }));

  return (
    <div
      style={{
        background: "linear-gradient(135deg, rgba(10,22,40,0.95), rgba(5,10,25,0.98))",
        border: "1px solid #0e7490",
        borderRadius: "14px",
        padding: "1.5rem",
        fontFamily: "var(--font-sans, sans-serif)",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: "1.25rem" }}>
        <h3 style={{ margin: 0, color: "#e2e8f0", fontSize: "1.05rem", fontWeight: 700 }}>
          🧮 Interactive VRAM Simulator
        </h3>
        <p style={{ margin: "0.3rem 0 0", fontSize: "0.78rem", color: "#64748b" }}>
          Toggle components to see which GPU can hold the full AOB stack. Required components are locked.
        </p>
      </div>

      {/* Component chips */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1.5rem" }}>
        {COMPONENTS.map((comp) => {
          const on = enabled[comp.id];
          return (
            <button
              key={comp.id}
              onClick={() => toggle(comp.id)}
              title={comp.description}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                padding: "0.35rem 0.75rem",
                borderRadius: "20px",
                border: `1px solid ${on ? comp.color : "rgba(148,163,184,0.2)"}`,
                background: on ? `${comp.color}18` : "rgba(255,255,255,0.03)",
                color: on ? comp.color : "#475569",
                fontSize: "0.75rem",
                fontWeight: 600,
                cursor: comp.required ? "not-allowed" : "pointer",
                transition: "all 0.2s ease",
                opacity: comp.required ? 1 : undefined,
              }}
            >
              <span
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: on ? comp.color : "#374151",
                  flexShrink: 0,
                }}
              />
              {comp.label}
              <span style={{ fontSize: "0.7rem", color: on ? `${comp.color}90` : "#374151" }}>
                {comp.gb} GB
              </span>
              {comp.required && (
                <span style={{ fontSize: "0.6rem", color: "#64748b" }}>🔒</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Dual bars */}
      <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
        <Bar
          used={totalGb}
          total={MI300X_TOTAL_GB}
          limit={MI300X_TOTAL_GB}
          label="AMD MI300X · 192 GB HBM3"
          oomLabel="192 GB"
          segments={mi300xSegments}
        />
        <Bar
          used={totalGb}
          total={MI300X_TOTAL_GB}
          limit={H100_LIMIT_GB}
          label="NVIDIA H100 · 80 GB limit"
          oomLabel="← 80 GB OOM threshold"
          segments={h100Segments}
        />
      </div>

      {/* Verdict */}
      <div
        style={{
          marginTop: "1.25rem",
          padding: "0.65rem 1rem",
          borderRadius: "8px",
          background: totalGb > H100_LIMIT_GB ? "rgba(239,68,68,0.08)" : "rgba(74,222,128,0.06)",
          border: `1px solid ${totalGb > H100_LIMIT_GB ? "rgba(239,68,68,0.3)" : "rgba(74,222,128,0.25)"}`,
          fontSize: "0.8rem",
          color: "var(--text-secondary)",
          lineHeight: 1.6,
        }}
      >
        {totalGb > H100_LIMIT_GB ? (
          <>
            <strong style={{ color: "#f87171" }}>H100 cannot run this configuration.</strong>{" "}
            Requires {totalGb.toFixed(0)} GB — {(totalGb - H100_LIMIT_GB).toFixed(0)} GB over the H100&apos;s 80 GB limit.
            The MI300X has{" "}
            <strong style={{ color: "#4ade80" }}>{(MI300X_TOTAL_GB - totalGb).toFixed(0)} GB headroom</strong> remaining.
          </>
        ) : (
          <>
            <strong style={{ color: "#4ade80" }}>Both GPUs could run this reduced configuration.</strong>{" "}
            Enable all components to see the MI300X advantage.
          </>
        )}
      </div>
    </div>
  );
}
