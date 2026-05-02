"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import NavBar from "@/components/NavBar";
import ConfidenceRing from "@/components/ConfidenceRing";
import BiomarkerPanel from "@/components/BiomarkerPanel";
import DebateTranscript from "@/components/DebateTranscript";
import HeatmapViewer from "@/components/HeatmapViewer";
import PfsChart from "@/components/PfsChart";
import BoardMemoryPanel from "@/components/BoardMemoryPanel";
import StatusBadge from "@/components/StatusBadge";
import { getReport, type BoardResult } from "@/lib/api";

function Section({ title, children, icon }: { title: string; children: React.ReactNode; icon?: string }) {
  return (
    <div className="glass-card" style={{ padding: "1.5rem" }}>
      <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "1.1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        {icon && <span>{icon}</span>}
        {title}
      </h3>
      {children}
    </div>
  );
}

export default function ReportPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const router = useRouter();
  const [result, setResult] = useState<BoardResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionNote, setActionNote] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const load = async () => {
      try {
        const r = await getReport(jobId);
        setResult(r);
      } catch (e) {
        setError("Failed to load report");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [jobId]);

  if (loading) {
    return (
      <>
        <NavBar />
        <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh", flexDirection: "column", gap: "1rem", color: "var(--text-muted)" }}>
          <div style={{ width: "40px", height: "40px", border: "3px solid rgba(13,148,136,0.2)", borderTopColor: "var(--teal-light)", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
          Loading report...
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      </>
    );
  }

  if (error || !result) {
    return (
      <>
        <NavBar />
        <div style={{ maxWidth: "600px", margin: "4rem auto", textAlign: "center", color: "var(--danger)", padding: "2rem" }}>
          <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>⚠️</div>
          <div>{error ?? "Report not found"}</div>
          <button className="btn-ghost" style={{ marginTop: "1.5rem", padding: "0.5rem 1.5rem" }} onClick={() => router.push("/")}>← Back to Home</button>
        </div>
      </>
    );
  }

  const plan = result.management_plan;
  const pathology = result.pathology_report;
  const debates = result.debate_rounds ?? plan?.debate_transcript ?? [];
  const similar = result.similar_cases ?? [];

  const handleShare = async () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      setActionNote("Copy not supported in this browser.");
      return;
    }
    try {
      await navigator.clipboard.writeText(window.location.href);
      setActionNote("Link copied to clipboard.");
    } catch (e) {
      setActionNote("Failed to copy link.");
    }
    setTimeout(() => setActionNote(null), 2000);
  };

  return (
    <>
      <NavBar />
      <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "2.5rem 2rem 6rem" }}>

        {/* Report header */}
        <div style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <h1 style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "0.3rem" }}>
                Clinical Report
              </h1>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                Case: {result.case_id} · Generated: {plan?.generated_at ? new Date(plan.generated_at).toLocaleString() : "—"}
              </p>
            </div>
            <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
              <StatusBadge status={result.status} />
              {result.total_time_s && (
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>⏱ {result.total_time_s.toFixed(1)}s</span>
              )}
            </div>
          </div>

          {/* Debate summary banner */}
          {debates.length > 0 && (
            <div
              style={{
                marginTop: "1rem",
                padding: "0.75rem 1.25rem",
                background: "rgba(13,148,136,0.08)",
                border: "1px solid var(--teal-border)",
                borderRadius: "8px",
                fontSize: "0.83rem",
                color: "var(--text-secondary)",
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
              }}
            >
              <span>⚖️</span>
              <span>
                <strong style={{ color: "var(--teal-light)" }}>{debates.length} debate round{debates.length !== 1 ? "s" : ""}</strong> conducted.
                Final consensus score:{" "}
                <strong style={{ color: "var(--success)" }}>
                  {plan?.consensus_score ?? debates[debates.length - 1]?.consensus_score ?? "—"}/100
                </strong>
              </span>
            </div>
          )}
        </div>

        {/* Two-column layout */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem" }}>

          {/* LEFT column */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>

            {/* Diagnosis */}
            {plan && (
              <Section title="Diagnosis" icon="🔬">
                <div style={{ display: "flex", gap: "1.25rem", alignItems: "flex-start" }}>
                  <ConfidenceRing value={plan.diagnosis.confidence} size={110} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 700, fontSize: "1.05rem", color: "var(--teal-light)", marginBottom: "0.35rem" }}>
                      {plan.diagnosis.primary}
                    </div>
                    <div style={{ marginBottom: "0.5rem" }}>
                      <div style={{ fontSize: "0.83rem", color: "var(--text-secondary)" }}>{plan.diagnosis.tnm_stage}</div>
                      <span
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          marginTop: "0.35rem",
                          padding: "0.18rem 0.5rem",
                          borderRadius: "999px",
                          background: "rgba(15, 23, 42, 0.55)", // charcoal
                          border: "1px solid rgba(13, 148, 136, 0.35)", // dark teal
                          color: "rgba(153, 246, 228, 0.95)",
                          fontSize: "0.68rem",
                          fontWeight: 600,
                          letterSpacing: "0.02em",
                          lineHeight: 1.2,
                        }}
                      >
                        Staging generated by custom Llama-3-8B-LoRA Specialist
                      </span>
                    </div>
                    {pathology?.uncertainty_interval && (
                      <div
                        style={{
                          fontSize: "0.75rem",
                          color: pathology.high_uncertainty ? "var(--warning)" : "var(--text-muted)",
                          background: pathology.high_uncertainty ? "rgba(245,158,11,0.08)" : "transparent",
                          borderRadius: "4px",
                          padding: pathology.high_uncertainty ? "0.2rem 0.4rem" : "0",
                        }}
                      >
                        {pathology.high_uncertainty ? "⚠️ " : ""}MC Dropout: {pathology.uncertainty_interval}
                      </div>
                    )}
                  </div>
                </div>
                {plan.patient_summary && (
                  <p style={{ fontSize: "0.83rem", color: "var(--text-secondary)", lineHeight: 1.6, marginTop: "1rem", paddingTop: "0.75rem", borderTop: "1px solid var(--border)" }}>
                    {plan.patient_summary}
                  </p>
                )}
              </Section>
            )}

            {/* Biomarkers */}
            {pathology?.biomarkers && Object.keys(pathology.biomarkers).length > 0 && (
              <Section title="Biomarker Profile" icon="🧬">
                <BiomarkerPanel biomarkers={pathology.biomarkers} />
              </Section>
            )}

            {/* Treatment plan */}
            {plan && (
              <Section title="Treatment Plan" icon="💊">
                <div
                  style={{
                    padding: "1rem",
                    background: "rgba(13,148,136,0.07)",
                    border: "1px solid var(--teal-border)",
                    borderRadius: "8px",
                    marginBottom: "1rem",
                  }}
                >
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600, marginBottom: "0.3rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>First-line</div>
                  <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--teal-light)", lineHeight: 1.4 }}>
                    {plan.treatment_plan.first_line}
                  </div>
                </div>
                {plan.treatment_plan.rationale && (
                  <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: "0.75rem" }}>
                    {plan.treatment_plan.rationale}
                  </p>
                )}
                {plan.treatment_plan.alternatives?.length > 0 && (
                  <div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, marginBottom: "0.35rem" }}>Alternatives</div>
                    {plan.treatment_plan.alternatives.map((a, i) => (
                      <div key={i} style={{ fontSize: "0.8rem", color: "var(--text-secondary)", padding: "0.25rem 0", borderTop: i === 0 ? "1px solid var(--border)" : "none" }}>
                        {i + 1}. {a}
                      </div>
                    ))}
                  </div>
                )}
              </Section>
            )}

            {/* Immediate actions */}
            {plan && plan.immediate_actions?.length > 0 && (
              <Section title="Immediate Actions" icon="⚡">
                <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                  {plan.immediate_actions.map((a, i) => (
                    <div key={i} style={{ display: "flex", gap: "0.5rem", fontSize: "0.83rem", color: "var(--text-secondary)" }}>
                      <span style={{ color: "var(--teal-light)", fontWeight: 700, flexShrink: 0 }}>{i + 1}.</span>
                      {a}
                    </div>
                  ))}
                </div>
              </Section>
            )}
          </div>

          {/* RIGHT column */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>

            {/* Heatmaps */}
            <Section title="Attention Heatmaps" icon="🗺️">
              <HeatmapViewer jobId={jobId} />
            </Section>

            {/* Debate transcript */}
            {debates.length > 0 && (
              <Section title="Agent Debate Transcript" icon="⚖️">
                <DebateTranscript rounds={debates} />
              </Section>
            )}

            {/* PFS chart */}
            {plan && (plan.pfs_curve?.length > 0 || plan.pfs_12mo) && (
              <Section title="Digital Twin — 12-month PFS" icon="📊">
                <PfsChart data={plan.pfs_curve} pfs12mo={plan.pfs_12mo} />
              </Section>
            )}

            {/* Board memory */}
            {similar.length > 0 && (
              <Section title="Board Memory — Similar Cases" icon="🧠">
                <BoardMemoryPanel cases={similar} />
              </Section>
            )}

            {/* Further investigations */}
            {plan && plan.further_investigations?.length > 0 && (
              <Section title="Further Investigations" icon="🔍">
                <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                  {plan.further_investigations.map((inv, i) => (
                    <div key={i} style={{ fontSize: "0.83rem", color: "var(--text-secondary)", display: "flex", gap: "0.5rem" }}>
                      <span style={{ color: "var(--text-muted)" }}>•</span> {inv}
                    </div>
                  ))}
                </div>
              </Section>
            )}

            {/* Citations */}
            {plan && plan.citations?.length > 0 && (
              <Section title="Citations" icon="📚">
                <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
                  {plan.citations.map((c, i) => (
                    <div key={i} style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                      [{i + 1}] {c}
                    </div>
                  ))}
                </div>
              </Section>
            )}
          </div>
        </div>

        {/* Board consensus */}
        {plan?.board_consensus && (
          <div
            className="glass-card"
            style={{ padding: "1.25rem", marginTop: "1.25rem", borderColor: "rgba(34,197,94,0.3)" }}
          >
            <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--success)", marginBottom: "0.35rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              ✅ Board Consensus
            </div>
            <p style={{ fontSize: "0.87rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              {plan.board_consensus}
            </p>
          </div>
        )}

        {/* Disclaimer */}
        <div style={{ marginTop: "1.5rem", padding: "0.75rem 1rem", background: "rgba(239,68,68,0.05)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: "8px", fontSize: "0.75rem", color: "rgba(239,68,68,0.7)" }}>
          ⚠️ {plan?.disclaimer ?? "AI research tool — NOT for clinical use. Always consult a qualified oncologist."}
        </div>

      </main>

      {/* Sticky actions */}
      <div
        style={{
          position: "fixed",
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(10,22,40,0.92)",
          borderTop: "1px solid var(--border-teal)",
          backdropFilter: "blur(12px)",
          zIndex: 50,
        }}
      >
        <div
          style={{
            maxWidth: "1200px",
            margin: "0 auto",
            padding: "0.75rem 2rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
            flexWrap: "wrap",
          }}
        >
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            {actionNote ?? "Share or export this report."}
          </div>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <button className="btn-ghost" style={{ padding: "0.5rem 1.25rem", fontSize: "0.85rem" }} onClick={handleShare}>
              Share
            </button>
            <button
              className="btn-teal"
              style={{ padding: "0.5rem 1.25rem", fontSize: "0.85rem" }}
              onClick={() => window.print()}
            >
              Export PDF
            </button>
            <button className="btn-ghost" style={{ padding: "0.5rem 1.25rem", fontSize: "0.85rem" }} onClick={() => router.push("/")}>
              Run Another Case
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
