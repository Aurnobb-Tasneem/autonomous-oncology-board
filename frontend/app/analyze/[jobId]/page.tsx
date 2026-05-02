"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import NavBar from "@/components/NavBar";
import AgentTimeline from "@/components/AgentTimeline";
import VramBar from "@/components/VramBar";
import StatusBadge from "@/components/StatusBadge";
import { streamJob, type SseStep, type JobStatus } from "@/lib/api";

const AGENT_META: Record<string, { icon: string; label: string; color: string }> = {
  pathologist: { icon: "🔬", label: "Pathologist", color: "#0d9488" },
  researcher: { icon: "📚", label: "Researcher", color: "#7c3aed" },
  oncologist: { icon: "👨‍⚕️", label: "Oncologist", color: "#0891b2" },
};

function debateHeuristic(step: SseStep): boolean {
  const msg = (step.message ?? "").toLowerCase();
  const agent = (step.agent ?? "").toLowerCase();
  const type = (step.type ?? "").toLowerCase();

  if (type.includes("debate") || type.includes("challenge") || type.includes("revise")) return true;

  // Backend emits debate traffic under researcher/oncologist/system — bucket it for the Debate panel.
  if (
    msg.includes("challenge") ||
    msg.includes("revis") ||
    msg.includes("consensus score") ||
    msg.includes("debate round") ||
    msg.includes("agent debate") ||
    msg.includes("round 1:") ||
    msg.includes("round 2:") ||
    msg.includes("round 3:")
  ) {
    if (agent.includes("research") || agent.includes("oncolog") || agent.includes("system")) return true;
  }

  return false;
}

function resolveAgent(step: SseStep): string {
  if (debateHeuristic(step)) return "debate";

  if (step.agent) {
    const a = step.agent.toLowerCase();
    if (a.includes("patholog") || a.includes("gigapath")) return "pathologist";
    if (a.includes("research")) return "researcher";
    if (a.includes("oncolog")) return "oncologist";
    // Qwen / VLM second opinion
    if (
      a.includes("qwen") ||
      a.includes("vlm_pathologist") ||
      a.includes("vlm-pathologist") ||
      a.includes("second_opinion")
    ) {
      return "pathologist"; // snapshot strip only has 3 tiles — keep VLM updates visible under Pathologist
    }
    return step.agent;
  }

  const type = step.type?.toLowerCase() ?? "";
  if (type.includes("patholog") || type.includes("gigapath")) return "pathologist";
  if (type.includes("research")) return "researcher";
  if (type.includes("oncolog")) return "oncologist";
  if (type.includes("qwen") || type.includes("vlm_pathologist") || type.includes("vlm")) return "pathologist";
  return "system";
}

export default function AnalyzePage() {
  const { jobId } = useParams<{ jobId: string }>();
  const router = useRouter();
  const [steps, setSteps] = useState<SseStep[]>([]);
  const [status, setStatus] = useState<JobStatus>("queued");
  const [elapsed, setElapsed] = useState(0);
  const [redirectIn, setRedirectIn] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startTime = useRef(Date.now());

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime.current) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!jobId) return;
    setStatus("running");

    const cleanup = streamJob(
      jobId,
      (step) => {
        setSteps((prev) => [...prev, step]);
      },
      () => {
        // Done
        setStatus("done");
        setRedirectIn(3);
      },
      (err) => {
        setError(err);
        setStatus("failed");
      }
    );

    return cleanup;
  }, [jobId]);

  // Countdown redirect
  useEffect(() => {
    if (redirectIn === null) return;
    if (redirectIn <= 0) {
      router.push(`/report/${jobId}`);
      return;
    }
    const t = setTimeout(() => setRedirectIn((n) => (n ?? 1) - 1), 1000);
    return () => clearTimeout(t);
  }, [redirectIn, jobId, router]);

  // Compute progress (backend sends AgentStep.progress 0–100)
  const lastProgress = [...steps].reverse().find((s) => typeof s.progress === "number")?.progress;
  const progress =
    status === "done"
      ? 100
      : typeof lastProgress === "number"
        ? Math.max(0, Math.min(100, Math.round(lastProgress)))
        : Math.min(Math.round((steps.length / 30) * 100), 95);

  const getLatestStep = (agent: string) => {
    for (let i = steps.length - 1; i >= 0; i -= 1) {
      if (resolveAgent(steps[i]) === agent) return steps[i];
    }
    return null;
  };

  const debateSteps = steps.filter((s) => resolveAgent(s) === "debate");

  return (
    <>
      <NavBar />
      <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "2.5rem 2rem" }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 340px), 1fr))",
            gap: "1.5rem",
            alignItems: "start",
          }}
        >
          <div style={{ minWidth: 0 }}>
            {/* Header */}
            <div style={{ marginBottom: "2rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
                <div>
                  <h1 style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "0.3rem" }}>
                    {status === "done" ? "✅ Analysis Complete" : "🔄 Analysis Running"}
                  </h1>
                  <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                    Job: {jobId}
                  </p>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>⏱ {elapsed}s</span>
                  <StatusBadge status={status} />
                </div>
              </div>

              {/* Progress bar */}
              <div style={{ marginTop: "1.25rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
                  <span>{steps.length} steps completed</span>
                  <span>{progress}%</span>
                </div>
                <div style={{ height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px", overflow: "hidden" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${progress}%`,
                      background: status === "done" ? "var(--success)" : "linear-gradient(90deg, var(--teal), var(--teal-light))",
                      borderRadius: "3px",
                      transition: "width 0.5s ease",
                      boxShadow: "0 0 8px rgba(13,148,136,0.5)",
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div
                className="glass-card"
                style={{
                  padding: "1rem 1.25rem",
                  borderColor: "var(--danger)",
                  color: "var(--danger)",
                  marginBottom: "1.5rem",
                }}
              >
                ⚠️ {error}
              </div>
            )}

            {/* Redirect notice */}
            {redirectIn !== null && (
              <div
                className="glass-card"
                style={{
                  padding: "1rem 1.25rem",
                  borderColor: "rgba(34,197,94,0.4)",
                  color: "var(--success)",
                  marginBottom: "1.5rem",
                  textAlign: "center",
                  fontWeight: 600,
                }}
              >
                ✅ Pipeline complete! Redirecting to report in {redirectIn}s...
              </div>
            )}

            {/* Agent snapshot */}
            <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.25rem" }}>
              <h2 style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginBottom: "1.25rem" }}>
                Live Agent Snapshot
              </h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
                {(["pathologist", "researcher", "oncologist"] as const).map((agent) => {
                  const meta = AGENT_META[agent];
                  const latest = getLatestStep(agent);
                  const active = Boolean(latest);
                  return (
                    <div
                      key={agent}
                      className="glass-card-sm"
                      style={{ padding: "1rem", borderColor: active ? meta.color : "var(--border)" }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
                        <span style={{ fontSize: "1.2rem" }}>{meta.icon}</span>
                        <div style={{ fontWeight: 700, fontSize: "0.9rem", color: meta.color }}>{meta.label}</div>
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
                        {latest ? "Latest update" : "Waiting for output"}
                      </div>
                      <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.5, minHeight: "2.8rem" }}>
                        {latest?.message ?? "Model loading and preparing..."}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Debate feed */}
            <div className="glass-card" style={{ padding: "1.5rem", marginBottom: "1.25rem" }}>
              <h2 style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginBottom: "1.25rem" }}>
                Debate Feed
              </h2>
              {debateSteps.length === 0 ? (
                <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                  Waiting for debate rounds...
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  {[...debateSteps].reverse().map((step, idx) => (
                    <div
                      key={idx}
                      className="glass-card-sm"
                      style={{ padding: "0.75rem 1rem", borderColor: "rgba(217,119,6,0.3)" }}
                    >
                      <div style={{ fontSize: "0.7rem", color: "#d97706", fontWeight: 700, marginBottom: "0.2rem" }}>
                        {step.type}
                      </div>
                      <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                        {step.message}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Timeline */}
            <div className="glass-card" style={{ padding: "1.75rem" }}>
              <h2 style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginBottom: "1.25rem" }}>
                Live Agent Pipeline
              </h2>
              <AgentTimeline steps={steps} elapsed={elapsed} />
            </div>

            {/* Manual report link */}
            {status === "done" && (
              <div style={{ textAlign: "center", marginTop: "1.5rem" }}>
                <button
                  className="btn-teal"
                  style={{ padding: "0.75rem 2rem", fontSize: "0.95rem" }}
                  onClick={() => router.push(`/report/${jobId}`)}
                >
                  View Full Report →
                </button>
              </div>
            )}

            {/* Still running hint */}
            {status === "running" && steps.length === 0 && (
              <div style={{ textAlign: "center", marginTop: "2rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                <div
                  style={{
                    width: "32px", height: "32px",
                    border: "3px solid rgba(13,148,136,0.2)",
                    borderTopColor: "var(--teal-light)",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                    margin: "0 auto 1rem",
                  }}
                />
                Loading models and preparing analysis…
              </div>
            )}
          </div>

          <aside style={{ position: "sticky", top: "80px", justifySelf: "stretch" }}>
            <VramBar />
            <p style={{ marginTop: "0.75rem", fontSize: "0.72rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
              Polls <span style={{ fontFamily: "monospace" }}>/api/vram</span> (rocm-smi JSON on the MI300X host). If you run the UI away from the API origin, point{" "}
              <span style={{ fontFamily: "monospace" }}>NEXT_PUBLIC_API_URL</span> at the FastAPI server.
            </p>
          </aside>
        </div>
      </main>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </>
  );
}
