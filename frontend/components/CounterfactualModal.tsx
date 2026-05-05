"use client";

import { useState } from "react";
import { runCounterfactual, type CounterfactualResponse } from "@/lib/api";
import RevisionDiff from "@/components/RevisionDiff";

interface CounterfactualModalProps {
  jobId: string;
  treatmentText: string;
  /** A preset hypothesis label for the trigger button. e.g. "What if EGFR negative?" */
  hypothesis: string;
}

export default function CounterfactualModal({
  jobId,
  treatmentText,
  hypothesis,
}: CounterfactualModalProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CounterfactualResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleOpen = async () => {
    setOpen(true);
    if (result) return; // Already loaded
    setLoading(true);
    setError(null);
    try {
      const resp = await runCounterfactual(jobId, hypothesis);
      setResult(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Counterfactual failed");
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => setOpen(false);

  return (
    <>
      {/* Trigger button */}
      <button
        onClick={handleOpen}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "0.35rem",
          padding: "0.2rem 0.6rem",
          borderRadius: "6px",
          fontSize: "0.72rem",
          fontWeight: 600,
          background: "rgba(124,58,237,0.12)",
          border: "1px solid rgba(124,58,237,0.35)",
          color: "#a78bfa",
          cursor: "pointer",
          transition: "all 0.2s ease",
          verticalAlign: "middle",
          marginLeft: "0.5rem",
        }}
        title={`Run counterfactual: ${hypothesis}`}
      >
        🔀 What if?
      </button>

      {/* Modal overlay */}
      {open && (
        <div
          onClick={(e) => e.target === e.currentTarget && handleClose()}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.75)",
            backdropFilter: "blur(6px)",
            zIndex: 100,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "1.5rem",
          }}
        >
          <div
            className="glass-card"
            style={{
              width: "100%",
              maxWidth: "700px",
              maxHeight: "80vh",
              overflowY: "auto",
              padding: "1.75rem",
              borderColor: "rgba(124,58,237,0.4)",
              boxShadow: "0 0 60px rgba(124,58,237,0.2)",
            }}
          >
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem" }}>
              <div>
                <div style={{ fontSize: "0.7rem", color: "#a78bfa", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.3rem" }}>
                  🔀 Counterfactual Analysis
                </div>
                <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  {hypothesis}
                </h2>
              </div>
              <button
                onClick={handleClose}
                style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: "1.2rem", padding: "0.25rem", lineHeight: 1 }}
              >
                ×
              </button>
            </div>

            {/* Loading */}
            {loading && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "column",
                  gap: "0.75rem",
                  padding: "2rem",
                  color: "var(--text-muted)",
                  fontSize: "0.85rem",
                }}
              >
                <div
                  style={{
                    width: "28px",
                    height: "28px",
                    border: "3px solid rgba(124,58,237,0.2)",
                    borderTopColor: "#a78bfa",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                  }}
                />
                Generating counterfactual plan…
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </div>
            )}

            {/* Error */}
            {error && (
              <div
                style={{
                  padding: "0.75rem 1rem",
                  background: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.3)",
                  borderRadius: "8px",
                  color: "#fca5a5",
                  fontSize: "0.83rem",
                }}
              >
                ⚠️ {error}
                <div style={{ marginTop: "0.5rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  The counterfactual endpoint may not be available in this deployment.
                </div>
              </div>
            )}

            {/* Result */}
            {result && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                {/* Original treatment */}
                <div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.5rem" }}>
                    Original Plan
                  </div>
                  <div
                    style={{
                      padding: "0.65rem 0.9rem",
                      background: "rgba(5,10,25,0.6)",
                      border: "1px solid rgba(148,163,184,0.12)",
                      borderRadius: "8px",
                      fontSize: "0.83rem",
                      color: "var(--text-secondary)",
                      lineHeight: 1.6,
                    }}
                  >
                    {result.original_first_line}
                  </div>
                </div>

                {/* Revised plan */}
                <div>
                  <div style={{ fontSize: "0.7rem", color: "#a78bfa", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.5rem" }}>
                    Revised Under: <em style={{ fontStyle: "normal", color: "var(--text-primary)" }}>{hypothesis}</em>
                  </div>
                  {result.revised_plan.treatment_plan?.first_line && (
                    <RevisionDiff
                      before={result.original_first_line}
                      after={result.revised_plan.treatment_plan.first_line}
                      label="What changed"
                    />
                  )}
                </div>

                {/* Summary */}
                {result.diff_summary && (
                  <div
                    style={{
                      padding: "0.75rem 1rem",
                      background: "rgba(124,58,237,0.06)",
                      border: "1px solid rgba(124,58,237,0.2)",
                      borderRadius: "8px",
                      fontSize: "0.8rem",
                      color: "var(--text-secondary)",
                      lineHeight: 1.6,
                    }}
                  >
                    <strong style={{ color: "#a78bfa" }}>Board Note:</strong> {result.diff_summary}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
