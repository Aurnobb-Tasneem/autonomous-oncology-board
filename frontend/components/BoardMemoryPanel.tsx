"use client";

import { type SimilarCase, tissueLabel } from "@/lib/api";

export default function BoardMemoryPanel({ cases }: { cases: SimilarCase[] }) {
  if (!cases || cases.length === 0) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
        No similar cases in board memory yet.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
      {cases.map((c, i) => {
        const simPct = Math.round(c.similarity * 100);
        const simColor = simPct >= 80 ? "var(--teal-light)" : simPct >= 60 ? "var(--warning)" : "var(--text-muted)";
        return (
          <div
            key={i}
            className="glass-card-sm"
            style={{ padding: "0.75rem", display: "flex", gap: "0.75rem", alignItems: "flex-start" }}
          >
            {/* Similarity circle */}
            <div
              style={{
                width: "44px",
                height: "44px",
                borderRadius: "50%",
                border: `2px solid ${simColor}`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                fontSize: "0.78rem",
                fontWeight: 700,
                color: simColor,
              }}
            >
              {simPct}%
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.15rem" }}>
                {tissueLabel(c.tissue_type)}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.2rem" }}>
                {c.case_id}
              </div>
              {c.first_line_tx && (
                <div style={{ fontSize: "0.75rem", color: "var(--teal-light)" }}>
                  Tx: {c.first_line_tx.slice(0, 80)}{c.first_line_tx.length > 80 ? "…" : ""}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
