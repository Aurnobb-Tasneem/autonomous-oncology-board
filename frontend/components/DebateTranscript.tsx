"use client";

import { useState } from "react";
import { type DebateRound } from "@/lib/api";
import RevisionDiff from "@/components/RevisionDiff";

type TranscriptMessage = {
  round: number;
  speaker: string;
  message: string;
  revised_first_line?: string;
};

function normalizeSpeaker(raw: string): string {
  const s = (raw ?? "").toLowerCase().trim();
  if (!s) return "system";
  if (s.includes("qwen") || s.includes("qwen-vl") || s.includes("vlm") || s.includes("vlm_pathologist") || s.includes("second_opinion")) return "qwen_vl";
  if (s.includes("patholog") || s.includes("gigapath")) return "pathologist";
  if (s.includes("research")) return "researcher";
  if (s.includes("oncolog")) return "oncologist";
  return s;
}

function roundToMessages(round: DebateRound): TranscriptMessage[] {
  // New shape: { round, speaker, message }
  if (typeof round.speaker === "string" && typeof round.message === "string") {
    return [
      {
        round: round.round,
        speaker: normalizeSpeaker(round.speaker),
        message: round.message,
        revised_first_line: round.revised_first_line,
      },
    ];
  }

  // Legacy shape: 3 fixed text fields.
  const msgs: TranscriptMessage[] = [];
  if (round.researcher_challenge) {
    msgs.push({ round: round.round, speaker: "researcher", message: round.researcher_challenge });
  }
  if (round.pathologist_referee) {
    msgs.push({ round: round.round, speaker: "pathologist", message: round.pathologist_referee });
  }
  // Some backends used oncologist_revision; others used revision_notes only.
  if (round.oncologist_revision) {
    msgs.push({ round: round.round, speaker: "oncologist", message: round.oncologist_revision });
  } else if (round.revision_notes) {
    msgs.push({ round: round.round, speaker: "oncologist", message: round.revision_notes });
  }
  return msgs;
}

function bubbleStyle(speaker: string): { label: string; labelColor: string; bubble: React.CSSProperties } {
  const base: React.CSSProperties = {
    borderRadius: "10px",
    padding: "0.75rem 0.9rem",
    fontSize: "0.83rem",
    color: "var(--text-secondary)",
    lineHeight: 1.6,
    border: "1px solid rgba(148,163,184,0.14)",
    background: "rgba(13,20,40,0.55)",
  };

  if (speaker === "qwen_vl") {
    return {
      label: "Qwen-VL Visual Assessment",
      labelColor: "rgba(12, 92, 86, 1)",
      bubble: {
        ...base,
        border: "1px solid rgba(10, 100, 94, 0.55)", // subtle dark teal
        background: "rgba(7, 23, 31, 0.58)",
        boxShadow: "0 0 0 1px rgba(15,118,110,0.14), 0 10px 26px rgba(0,0,0,0.22)",
        backdropFilter: "blur(14px) saturate(1.25)",
      },
    };
  }

  if (speaker === "researcher") {
    return {
      label: "Researcher Challenge",
      labelColor: "#d97706",
      bubble: { ...base, background: "rgba(217,119,6,0.06)", border: "1px solid rgba(217,119,6,0.2)" },
    };
  }

  if (speaker === "pathologist") {
    return {
      label: "Pathologist Referee",
      labelColor: "var(--teal-light)",
      bubble: { ...base, background: "rgba(13,148,136,0.06)", border: "1px solid rgba(13,148,136,0.2)" },
    };
  }

  if (speaker === "oncologist") {
    return {
      label: "Oncologist Revision",
      labelColor: "var(--success)",
      bubble: { ...base, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.2)", color: "var(--success)", fontWeight: 500 },
    };
  }

  return { label: speaker.replace(/_/g, " ").toUpperCase(), labelColor: "var(--text-muted)", bubble: base };
}

export default function DebateTranscript({ rounds }: { rounds: DebateRound[] }) {
  const [openRound, setOpenRound] = useState<number | null>(0);

  if (!rounds || rounds.length === 0) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: "0.85rem", padding: "0.75rem 0" }}>
        No debate rounds recorded.
      </div>
    );
  }

  // Group messages by round number (supports both legacy and transcript formats).
  const grouped = rounds.reduce((acc, r) => {
    const msgs = roundToMessages(r);
    const k = r.round;
    const arr = acc.get(k) ?? [];
    acc.set(k, [...arr, ...msgs]);
    return acc;
  }, new Map<number, TranscriptMessage[]>());

  const orderedRounds = [...grouped.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([roundNum, messages]) => {
      // Stable ordering inside a round for readability.
      const order = { researcher: 1, qwen_vl: 2, pathologist: 3, oncologist: 4 };
      const sorted = [...messages].sort((m1, m2) => (order[m1.speaker as keyof typeof order] ?? 99) - (order[m2.speaker as keyof typeof order] ?? 99));
      return { roundNum, messages: sorted };
    });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {orderedRounds.map(({ roundNum, messages }, idx) => {
        const isOpen = openRound === idx;
        const rawScore = rounds.find((r) => r.round === roundNum)?.consensus_score;
        const score = typeof rawScore === "number" ? rawScore : null;
        const passed = score !== null ? score >= 70 : true;
        return (
          <div
            key={idx}
            className="glass-card-sm"
            style={{
              overflow: "hidden",
              borderColor: passed ? "rgba(34,197,94,0.3)" : "rgba(245,158,11,0.3)",
            }}
          >
            {/* Accordion header */}
            <button
              onClick={() => setOpenRound(isOpen ? null : idx)}
              style={{
                width: "100%",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "0.85rem 1rem",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                color: "var(--text-primary)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.65rem" }}>
                <span style={{ fontSize: "1rem" }}>⚖️</span>
                <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>Debate Round {roundNum}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.65rem" }}>
                {score !== null && (
                  <span
                    style={{
                      background: passed ? "rgba(34,197,94,0.12)" : "rgba(245,158,11,0.12)",
                      border: `1px solid ${passed ? "rgba(34,197,94,0.4)" : "rgba(245,158,11,0.4)"}`,
                      color: passed ? "var(--success)" : "var(--warning)",
                      padding: "0.15rem 0.6rem",
                      borderRadius: "20px",
                      fontSize: "0.78rem",
                      fontWeight: 700,
                    }}
                  >
                    {passed ? "✅" : "🔄"} Score {score}/100
                  </span>
                )}
                <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>{isOpen ? "▲" : "▼"}</span>
              </div>
            </button>

            {/* Accordion body */}
            {isOpen && (
              <div
                style={{
                  padding: "0 1rem 1rem",
                  borderTop: "1px solid var(--border)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.85rem",
                }}
              >
                {messages.map((m, mi) => {
                  const cfg = bubbleStyle(m.speaker);
                  return (
                    <div key={`${roundNum}-${mi}`}>
                      <div
                        style={{
                          fontSize: "0.72rem",
                          color: cfg.labelColor,
                          fontWeight: 700,
                          marginBottom: "0.3rem",
                          marginTop: mi === 0 ? "0.75rem" : undefined,
                        }}
                      >
                        {m.speaker === "qwen_vl" ? "🧠 " : m.speaker === "researcher" ? "📚 " : m.speaker === "pathologist" ? "🔬 " : m.speaker === "oncologist" ? "👨‍⚕️ " : "💬 "}
                        {cfg.label.toUpperCase()}
                      </div>
                      <div style={cfg.bubble}>
                        {m.message}
                        {m.revised_first_line && (
                          <div style={{ marginTop: "0.6rem", paddingTop: "0.6rem", borderTop: "1px solid rgba(148,163,184,0.12)" }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: "0.35rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                              First-line treatment revised
                            </div>
                            <RevisionDiff
                              before={m.message.split(".")[0] ?? ""}
                              after={m.revised_first_line}
                              label="Revision diff"
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
