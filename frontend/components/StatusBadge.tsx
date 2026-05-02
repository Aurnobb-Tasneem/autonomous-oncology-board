"use client";

import { type JobStatus } from "@/lib/api";

const CONFIGS: Record<
  JobStatus,
  { label: string; bg: string; color: string; border: string; dot: string }
> = {
  queued: {
    label: "Queued",
    bg: "rgba(245,158,11,0.1)",
    color: "#f59e0b",
    border: "rgba(245,158,11,0.4)",
    dot: "#f59e0b",
  },
  running: {
    label: "Running",
    bg: "rgba(13,148,136,0.12)",
    color: "var(--teal-light)",
    border: "var(--teal-border)",
    dot: "var(--teal-light)",
  },
  done: {
    label: "Done",
    bg: "rgba(34,197,94,0.1)",
    color: "var(--success)",
    border: "rgba(34,197,94,0.4)",
    dot: "var(--success)",
  },
  failed: {
    label: "Failed",
    bg: "rgba(239,68,68,0.1)",
    color: "var(--danger)",
    border: "rgba(239,68,68,0.4)",
    dot: "var(--danger)",
  },
};

export default function StatusBadge({ status }: { status: JobStatus }) {
  const cfg = CONFIGS[status];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.35rem",
        background: cfg.bg,
        border: `1px solid ${cfg.border}`,
        color: cfg.color,
        padding: "0.2rem 0.65rem",
        borderRadius: "20px",
        fontSize: "0.78rem",
        fontWeight: 600,
      }}
    >
      <span
        style={{
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          background: cfg.dot,
          animation: status === "running" ? "pulse-dot 1.8s ease-in-out infinite" : "none",
        }}
      />
      {cfg.label}
    </span>
  );
}
