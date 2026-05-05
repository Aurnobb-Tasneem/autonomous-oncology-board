"use client";

import { type TrainingReport } from "@/lib/api";

export interface SpecialistSpec {
  id: string;
  name: string;
  subtitle: string;
  icon: string;
  color: string;
  badge: string;
  params: string;
  vram: string;
  precision: string;
  source: string;
  sourceUrl?: string;
  description: string;
  sampleInput: string;
  sampleOutput: string;
  liveStatus?: "loaded" | "skipped" | "unknown";
  trainingReport?: TrainingReport | null;
}

function MetaRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.3rem 0", borderBottom: "1px solid rgba(148,163,184,0.08)" }}>
      <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600 }}>{label}</span>
      <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", fontFamily: mono ? "monospace" : undefined }}>{value}</span>
    </div>
  );
}

export default function SpecialistCard({ spec }: { spec: SpecialistSpec }) {
  const statusColor =
    spec.liveStatus === "loaded"
      ? "#4ade80"
      : spec.liveStatus === "skipped"
      ? "#f59e0b"
      : "#64748b";
  const statusLabel =
    spec.liveStatus === "loaded"
      ? "● Loaded"
      : spec.liveStatus === "skipped"
      ? "◌ Skipped"
      : "◌ Status unknown";

  return (
    <div
      className="glass-card"
      style={{
        padding: "1.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        borderColor: `${spec.color}40`,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Accent glow */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: "3px",
          background: `linear-gradient(90deg, ${spec.color}80, ${spec.color}20)`,
        }}
      />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.75rem" }}>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <div
            style={{
              fontSize: "1.8rem",
              width: "2.5rem",
              height: "2.5rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: `${spec.color}15`,
              borderRadius: "10px",
              border: `1px solid ${spec.color}30`,
              flexShrink: 0,
            }}
          >
            {spec.icon}
          </div>
          <div>
            <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)" }}>{spec.name}</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{spec.subtitle}</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "0.3rem" }}>
          <span
            style={{
              padding: "0.2rem 0.55rem",
              borderRadius: "20px",
              fontSize: "0.68rem",
              fontWeight: 700,
              background: `${spec.color}20`,
              border: `1px solid ${spec.color}50`,
              color: spec.color,
              whiteSpace: "nowrap",
            }}
          >
            {spec.badge}
          </span>
          <span style={{ fontSize: "0.67rem", color: statusColor, fontWeight: 600 }}>{statusLabel}</span>
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
        {spec.description}
      </p>

      {/* Meta table */}
      <div>
        <MetaRow label="Parameters" value={spec.params} />
        <MetaRow label="VRAM" value={spec.vram} />
        <MetaRow label="Precision" value={spec.precision} mono />
        <MetaRow label="Source" value={spec.source} />
        {spec.trainingReport && (
          <>
            <MetaRow label="LoRA Rank / α" value={`${spec.trainingReport.rank} / ${spec.trainingReport.alpha}`} mono />
            <MetaRow label="Train loss" value={spec.trainingReport.train_loss.toFixed(4)} mono />
            {spec.trainingReport.eval_loss != null && (
              <MetaRow label="Eval loss" value={spec.trainingReport.eval_loss.toFixed(4)} mono />
            )}
            <MetaRow label="Steps" value={String(spec.trainingReport.total_steps)} mono />
          </>
        )}
      </div>

      {/* Sample I/O */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: "0.3rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>Sample Input</div>
          <div
            style={{
              background: "rgba(5,10,25,0.7)",
              border: "1px solid rgba(148,163,184,0.1)",
              borderRadius: "6px",
              padding: "0.5rem 0.75rem",
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              lineHeight: 1.5,
              fontFamily: "monospace",
            }}
          >
            {spec.sampleInput}
          </div>
        </div>
        <div>
          <div style={{ fontSize: "0.68rem", color: spec.color, fontWeight: 700, marginBottom: "0.3rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>Sample Output</div>
          <div
            style={{
              background: `${spec.color}08`,
              border: `1px solid ${spec.color}25`,
              borderRadius: "6px",
              padding: "0.5rem 0.75rem",
              fontSize: "0.75rem",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
            }}
          >
            {spec.sampleOutput}
          </div>
        </div>
      </div>
    </div>
  );
}
