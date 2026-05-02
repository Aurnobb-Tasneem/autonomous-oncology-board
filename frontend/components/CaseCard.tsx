"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { runDemoCase, type DemoCase } from "@/lib/api";

const TISSUE_ICONS: Record<string, string> = {
  lung_adenocarcinoma: "🫁",
  colon_adenocarcinoma: "🔬",
  lung_squamous_cell_carcinoma: "🫁",
};

const TISSUE_COLORS: Record<string, string> = {
  lung_adenocarcinoma: "var(--teal)",
  colon_adenocarcinoma: "#7c3aed",
  lung_squamous_cell_carcinoma: "#0891b2",
};

const EXPECTED_TIME: Record<string, string> = {
  lung_adenocarcinoma: "~60s · Debate included",
  colon_adenocarcinoma: "~55s · Debate included",
  lung_squamous_cell_carcinoma: "~75s · 2 debate rounds",
};

export default function CaseCard({ demo }: { demo: DemoCase }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const icon = TISSUE_ICONS[demo.tissue_type] ?? "🔬";
  const accentColor = TISSUE_COLORS[demo.tissue_type] ?? "var(--teal)";
  const expectedTime = EXPECTED_TIME[demo.tissue_type] ?? "~60s";

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await runDemoCase(demo.case_name);
      router.push(`/analyze/${resp.job_id}`);
    } catch (e) {
      setError("Failed to start — is the API running?");
      setLoading(false);
    }
  };

  return (
    <div
      className="glass-card"
      style={{
        padding: "1.4rem",
        cursor: loading ? "wait" : "pointer",
        transition: "all 0.25s ease",
        borderColor: loading ? accentColor : "var(--border-teal)",
        boxShadow: loading
          ? `0 0 30px rgba(13,148,136,0.3), 0 0 60px rgba(13,148,136,0.1)`
          : "0 0 20px var(--teal-glow)",
        transform: loading ? "scale(1.01)" : "scale(1)",
      }}
      onClick={!loading ? handleRun : undefined}
    >
      {/* Icon + tissue */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem", marginBottom: "0.75rem" }}>
        <div
          style={{
            fontSize: "1.8rem",
            lineHeight: 1,
            filter: `drop-shadow(0 0 8px ${accentColor})`,
          }}
        >
          {icon}
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: "0.95rem", color: "var(--text-primary)", marginBottom: "0.15rem" }}>
            {demo.tissue_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            {demo.description}
          </div>
        </div>
      </div>

      {/* Expected time */}
      <div
        style={{
          fontSize: "0.75rem",
          color: "var(--text-muted)",
          marginBottom: "1rem",
          display: "flex",
          alignItems: "center",
          gap: "0.35rem",
        }}
      >
        <span>⏱</span>
        <span>{expectedTime}</span>
      </div>

      {/* Button */}
      {error ? (
        <p style={{ fontSize: "0.78rem", color: "var(--danger)", textAlign: "center" }}>{error}</p>
      ) : (
        <button
          className="btn-teal"
          disabled={loading}
          style={{
            width: "100%",
            padding: "0.65rem",
            fontSize: "0.88rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.5rem",
            opacity: loading ? 0.8 : 1,
          }}
          onClick={(e) => { e.stopPropagation(); handleRun(); }}
        >
          {loading ? (
            <>
              <span
                style={{
                  width: "14px",
                  height: "14px",
                  border: "2px solid rgba(255,255,255,0.3)",
                  borderTopColor: "white",
                  borderRadius: "50%",
                  display: "inline-block",
                  animation: "spin 0.8s linear infinite",
                }}
              />
              Starting...
            </>
          ) : (
            <>▶ Run Demo Case</>
          )}
        </button>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
