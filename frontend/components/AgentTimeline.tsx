"use client";

import { type AgentId, type SseStep } from "@/lib/api";

interface AgentTimelineProps {
  steps: SseStep[];
  elapsed: number;
}

const AGENTS: AgentId[] = ["pathologist", "second_opinion", "researcher", "tnm_specialist", "oncologist"];

const ACTIVE_TEAL = "#0f766e";
const ACTIVE_GLASS_BG = "rgba(7, 23, 31, 0.55)";
const ACTIVE_GLASS_BORDER = "rgba(15, 118, 110, 0.45)";

const AGENT_LABELS: Record<AgentId, { icon: string; label: string; color: string }> = {
  pathologist: { icon: "🔬", label: "GigaPath Pathologist", color: "#0d9488" },
  second_opinion: { icon: "🧠", label: "Qwen-VL Second Opinion", color: "#22c55e" },
  researcher: { icon: "📚", label: "Researcher", color: "#7c3aed" },
  tnm_specialist: { icon: "🧬", label: "Llama 3.1 8B TNM (LoRA)", color: "#38bdf8" },
  oncologist: { icon: "👨‍⚕️", label: "Main Oncologist", color: "#0891b2" },
  debate: { icon: "⚖️", label: "Debate", color: "#d97706" },
  system: { icon: "⚙️", label: "System", color: "#64748b" },
};

function normalizeAgentId(raw: string): AgentId | null {
  const s = raw.toLowerCase().trim();
  if (!s) return null;

  if (
    s.includes("qwen") ||
    s.includes("qwen-vl") ||
    s.includes("qwen2.5") ||
    s.includes("vlm_pathologist") ||
    s.includes("vlm-pathologist") ||
    s.includes("vlm pathologist") ||
    s.includes("second_opinion") ||
    s.includes("second opinion") ||
    s.includes("second-opinion")
  )
    return "second_opinion";
  if (s.includes("tnm") || s.includes("tnm_specialist") || s.includes("tnm specialist") || s.includes("tnm-specialist") || s.includes("staging"))
    return "tnm_specialist";

  if (s.includes("patholog") || s.includes("gigapath")) return "pathologist";
  if (s.includes("research")) return "researcher";
  if (s.includes("oncolog")) return "oncologist";
  if (s.includes("debate") || s.includes("challenge") || s.includes("revise") || s.includes("critique")) return "debate";
  if (s.includes("system")) return "system";
  return null;
}

function getAgent(step: SseStep): AgentId {
  // Heuristic: if a step message is explicitly about TNM/staging, show it in the TNM column
  // even when the backend reports it under system/oncologist.
  const msg = (step.message ?? "").toLowerCase();
  if (
    msg.includes("tnm") ||
    msg.includes("staging") ||
    msg.includes("ajcc") ||
    msg.includes("stage i") ||
    msg.includes("stage ii") ||
    msg.includes("stage iii") ||
    msg.includes("stage iv")
  ) {
    return "tnm_specialist";
  }

  const fromAgent = typeof step.agent === "string" ? normalizeAgentId(step.agent) : null;
  if (fromAgent) return fromAgent;
  const fromType = normalizeAgentId(step.type ?? "");
  return fromType ?? "system";
}

export default function AgentTimeline({ steps, elapsed }: AgentTimelineProps) {
  const latestStep = steps.length > 0 ? steps[steps.length - 1] : null;
  const currentAgent: AgentId | null = latestStep ? getAgent(latestStep) : null;

  return (
    <div>
      {/* Agent column headers */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "1rem",
          marginBottom: "1rem",
        }}
      >
        {AGENTS.map((agent) => {
          const cfg = AGENT_LABELS[agent];
          const agentSteps = steps.filter((s) => getAgent(s) === agent);
          const active = agentSteps.length > 0;
          const isCurrent = currentAgent === agent;
          return (
            <div
              key={agent}
              className="glass-card-sm"
              style={{
                padding: "0.75rem",
                textAlign: "center",
                borderColor: isCurrent ? ACTIVE_TEAL : active ? cfg.color : "var(--border)",
                opacity: active || isCurrent ? 1 : 0.45,
                transition: "all 0.3s ease",
                background: isCurrent ? ACTIVE_GLASS_BG : undefined,
                boxShadow: isCurrent ? "0 0 0 1px rgba(15,118,110,0.18), 0 10px 30px rgba(0,0,0,0.22)" : undefined,
                backdropFilter: isCurrent ? "blur(14px) saturate(1.25)" : undefined,
              }}
            >
              <div style={{ fontSize: "1.5rem" }}>{cfg.icon}</div>
              <div
                style={{
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  color: isCurrent ? ACTIVE_TEAL : active ? cfg.color : "var(--text-muted)",
                  marginTop: "0.25rem",
                }}
              >
                {cfg.label}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "0.15rem" }}>
                {agentSteps.length} step{agentSteps.length !== 1 ? "s" : ""}
              </div>
            </div>
          );
        })}
      </div>

      {/* Step log */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "420px", overflowY: "auto" }}>
        {steps.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "2rem", fontSize: "0.88rem" }}>
            Waiting for pipeline to start...
          </div>
        )}
        {[...steps].reverse().map((step, i) => {
          const agent = getAgent(step);
          const cfg = AGENT_LABELS[agent] ?? AGENT_LABELS.system;
          const isCurrent = currentAgent === agent && agent !== "system";
          return (
            <div
              key={i}
              className="step-enter"
              style={{
                display: "flex",
                gap: "0.65rem",
                alignItems: "flex-start",
                padding: "0.65rem 0.75rem",
                background: isCurrent ? ACTIVE_GLASS_BG : "rgba(13,20,40,0.6)",
                borderRadius: "8px",
                border: `1px solid ${isCurrent ? ACTIVE_GLASS_BORDER : "rgba(148,163,184,0.14)"}`,
                boxShadow: isCurrent ? "0 0 0 1px rgba(15,118,110,0.12)" : undefined,
                backdropFilter: isCurrent ? "blur(14px) saturate(1.25)" : undefined,
              }}
            >
              <span style={{ fontSize: "1rem", flexShrink: 0 }}>{cfg.icon}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: "0.72rem", color: isCurrent ? ACTIVE_TEAL : cfg.color, fontWeight: 600, marginBottom: "0.15rem" }}>
                  {cfg.label} · {step.type}
                </div>
                <div
                  style={{
                    fontSize: "0.82rem",
                    color: "var(--text-secondary)",
                    lineHeight: 1.5,
                    wordBreak: "break-word",
                  }}
                >
                  {step.message}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Elapsed */}
      {elapsed > 0 && (
        <div style={{ textAlign: "right", fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
          ⏱ {elapsed}s elapsed
        </div>
      )}
    </div>
  );
}
