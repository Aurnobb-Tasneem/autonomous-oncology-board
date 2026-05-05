"use client";

import { useState, useEffect, useRef } from "react";
import { runConcurrent, streamJob, type SseStep, type JobStatus } from "@/lib/api";

interface JobState {
  jobId: string;
  caseLabel: string;
  steps: SseStep[];
  status: JobStatus;
  elapsed: number;
}

const DEFAULT_CASES = [
  "lung_adenocarcinoma",
  "colon_adenocarcinoma",
  "lung_squamous",
];

const CASE_COLORS: Record<string, string> = {
  lung_adenocarcinoma: "#0d9488",
  colon_adenocarcinoma: "#22c55e",
  lung_squamous: "#38bdf8",
};

function MiniTimeline({ job }: { job: JobState }) {
  const color = CASE_COLORS[job.caseLabel] ?? "#0d9488";
  const statusIcon =
    job.status === "done" ? "✅" : job.status === "failed" ? "❌" : job.status === "running" ? "🔄" : "⏳";

  return (
    <div
      className="glass-card"
      style={{ padding: "1rem", borderColor: `${color}40`, flex: 1, minWidth: "240px" }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
        <div>
          <div style={{ fontSize: "0.85rem", fontWeight: 700, color, textTransform: "capitalize" }}>
            {job.caseLabel.replace(/_/g, " ")}
          </div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", fontFamily: "monospace", marginTop: "0.1rem" }}>
            {job.jobId.slice(0, 12)}…
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span style={{ fontSize: "0.85rem" }}>{statusIcon}</span>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{job.elapsed}s</span>
        </div>
      </div>

      {/* Mini step log */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.3rem",
          maxHeight: "180px",
          overflowY: "auto",
        }}
      >
        {job.steps.length === 0 && (
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", padding: "0.5rem 0" }}>
            Waiting for pipeline…
          </div>
        )}
        {[...job.steps].reverse().slice(0, 8).map((step, i) => (
          <div
            key={i}
            style={{
              fontSize: "0.72rem",
              color: "var(--text-secondary)",
              padding: "0.25rem 0.4rem",
              background: "rgba(5,10,25,0.5)",
              borderRadius: "4px",
              lineHeight: 1.4,
              borderLeft: `2px solid ${color}60`,
            }}
          >
            {step.message.length > 80 ? step.message.slice(0, 77) + "…" : step.message}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ConcurrentRunner() {
  const [jobs, setJobs] = useState<JobState[]>([]);
  const [running, setRunning] = useState(false);
  const [peakVram, setPeakVram] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cleanupRefs = useRef<(() => void)[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRun = async () => {
    setRunning(true);
    setError(null);
    setJobs([]);
    setPeakVram(null);

    // Cancel any lingering SSE streams
    cleanupRefs.current.forEach((fn) => fn());
    cleanupRefs.current = [];

    try {
      const resp = await runConcurrent(DEFAULT_CASES);
      const initialJobs: JobState[] = resp.job_ids.map((id, i) => ({
        jobId: id,
        caseLabel: DEFAULT_CASES[i] ?? `case_${i}`,
        steps: [],
        status: "running",
        elapsed: 0,
      }));
      setJobs(initialJobs);

      const startTimes = initialJobs.map(() => Date.now());

      // Tick elapsed per job
      timerRef.current = setInterval(() => {
        setJobs((prev) =>
          prev.map((j, i) => ({
            ...j,
            elapsed: j.status === "running" ? Math.floor((Date.now() - startTimes[i]) / 1000) : j.elapsed,
          }))
        );
      }, 1000);

      // SSE for each job
      resp.job_ids.forEach((jobId, i) => {
        const cleanup = streamJob(
          jobId,
          (step) => {
            setJobs((prev) =>
              prev.map((j) => (j.jobId === jobId ? { ...j, steps: [...j.steps, step] } : j))
            );
          },
          () => {
            setJobs((prev) =>
              prev.map((j) => (j.jobId === jobId ? { ...j, status: "done" } : j))
            );
          },
          (err) => {
            setJobs((prev) =>
              prev.map((j) => (j.jobId === jobId ? { ...j, status: "failed" } : j))
            );
          }
        );
        cleanupRefs.current.push(cleanup);
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to launch concurrent run";
      setError(msg);
      setRunning(false);
    }
  };

  // Watch VRAM during run
  useEffect(() => {
    if (!running) return;
    const pollVram = async () => {
      try {
        const res = await fetch("/api/proxy/api/vram", { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (typeof data.used_gb === "number") {
            setPeakVram((prev) => Math.max(prev ?? 0, data.used_gb));
          }
        }
      } catch {
        // ignore
      }
    };
    const poll = setInterval(pollVram, 2000);
    return () => clearInterval(poll);
  }, [running]);

  // Check all done
  useEffect(() => {
    if (jobs.length > 0 && jobs.every((j) => j.status === "done" || j.status === "failed")) {
      setRunning(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, [jobs]);

  useEffect(() => {
    return () => {
      cleanupRefs.current.forEach((fn) => fn());
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const allDone = jobs.length > 0 && jobs.every((j) => j.status === "done" || j.status === "failed");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Controls */}
      <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
              3-Case Concurrent Stress Test
            </h2>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.3rem", marginBottom: 0 }}>
              Launches 3 cancer cases simultaneously on the MI300X. Watch VRAM climb.
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            {peakVram !== null && (
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "#22d3ee" }}>{peakVram.toFixed(1)} GB</div>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>peak VRAM</div>
              </div>
            )}
            <button
              className="btn-teal"
              style={{ padding: "0.6rem 1.5rem", fontSize: "0.9rem", opacity: running ? 0.6 : 1 }}
              onClick={startRun}
              disabled={running}
            >
              {running ? "Running…" : jobs.length > 0 ? "Run Again" : "Launch 3 Concurrent Cases"}
            </button>
          </div>
        </div>

        {/* Case chips */}
        <div style={{ display: "flex", gap: "0.6rem", marginTop: "1rem", flexWrap: "wrap" }}>
          {DEFAULT_CASES.map((c) => (
            <span
              key={c}
              style={{
                padding: "0.2rem 0.65rem",
                borderRadius: "20px",
                fontSize: "0.72rem",
                fontWeight: 600,
                background: `${CASE_COLORS[c]}15`,
                border: `1px solid ${CASE_COLORS[c]}40`,
                color: CASE_COLORS[c],
                textTransform: "capitalize",
              }}
            >
              {c.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          className="glass-card"
          style={{ padding: "1rem", borderColor: "rgba(239,68,68,0.4)", color: "var(--danger)", fontSize: "0.85rem" }}
        >
          ⚠️ {error}
        </div>
      )}

      {/* Job cards */}
      {jobs.length > 0 && (
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          {jobs.map((job) => (
            <MiniTimeline key={job.jobId} job={job} />
          ))}
        </div>
      )}

      {/* Results summary */}
      {allDone && (
        <div
          className="glass-card"
          style={{ padding: "1rem 1.25rem", borderColor: "rgba(34,197,94,0.3)" }}
        >
          <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--success)", marginBottom: "0.5rem" }}>
            ✅ All cases completed
          </div>
          <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
            {jobs.map((j) => (
              <div key={j.jobId} style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
                <span style={{ color: CASE_COLORS[j.caseLabel] ?? "var(--teal-light)", fontWeight: 600, textTransform: "capitalize" }}>
                  {j.caseLabel.replace(/_/g, " ")}
                </span>
                {" "}{j.status === "done" ? "✅" : "❌"} {j.elapsed}s · {j.steps.length} steps
              </div>
            ))}
          </div>
          {peakVram !== null && (
            <div style={{ marginTop: "0.5rem", fontSize: "0.78rem", color: "var(--text-muted)" }}>
              Peak VRAM: <strong style={{ color: "#22d3ee" }}>{peakVram.toFixed(1)} GB</strong>
              {peakVram > 80 && <span style={{ color: "#f87171", fontWeight: 700 }}> — H100 would OOM</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
