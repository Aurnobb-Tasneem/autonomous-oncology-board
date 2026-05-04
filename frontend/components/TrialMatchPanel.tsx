"use client";

interface TrialMatch {
  trial_id: string;
  title: string;
  phase: string;
  cancer_type: string;
  biomarker_focus: string;
  eligibility_score: number;
  eligibility_flags: Record<string, string>;
  nct_id: string;
  study_status: string;
  brief_summary: string;
  inclusion_snippet: string;
  exclusion_snippet: string;
  contact_info: string;
}

interface Props {
  trials: TrialMatch[];
}

function badge(score: number): { label: string; color: string; bg: string } {
  if (score >= 0.7)  return { label: "Likely eligible",       color: "#22d3ee", bg: "rgba(14,116,144,0.15)" };
  if (score >= 0.4)  return { label: "Potentially eligible",  color: "#f59e0b", bg: "rgba(245,158,11,0.1)"  };
  return               { label: "Criteria mismatch",         color: "#94a3b8", bg: "rgba(255,255,255,0.04)" };
}

function flagColor(status: string): string {
  if (status === "eligible") return "#22d3ee";
  if (status === "check")    return "#f59e0b";
  if (status === "excluded") return "#f87171";
  return "#64748b";
}

function phaseColor(phase: string): string {
  if (phase.includes("III") || phase.includes("IV")) return "#22d3ee";
  if (phase.includes("II"))  return "#a78bfa";
  return "#64748b";
}

export default function TrialMatchPanel({ trials }: Props) {
  if (!trials?.length) {
    return (
      <div
        style={{
          background: "rgba(10,22,40,0.85)",
          border: "1px solid #1e3a5f",
          borderRadius: 12,
          padding: "1.1rem 1.3rem",
          color: "#475569",
          fontSize: "0.82rem",
          fontFamily: "var(--font-sans, sans-serif)",
        }}
      >
        🧪 No matching clinical trials found in local corpus.
      </div>
    );
  }

  return (
    <div
      style={{
        background: "rgba(10,22,40,0.85)",
        border: "1px solid #1e3a5f",
        borderRadius: 12,
        padding: "1.1rem 1.3rem",
        fontFamily: "var(--font-sans, sans-serif)",
      }}
    >
      <h3
        style={{
          margin: "0 0 1rem 0",
          color: "#e2e8f0",
          fontSize: "0.95rem",
          fontWeight: 700,
        }}
      >
        🧪 Clinical Trial Matches
        <span
          style={{
            marginLeft: 8,
            fontSize: "0.72rem",
            color: "#64748b",
            fontWeight: 400,
          }}
        >
          Local ClinicalTrials.gov corpus · {trials.length} match
          {trials.length !== 1 ? "es" : ""}
        </span>
      </h3>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
        {trials.map((t, i) => {
          const b = badge(t.eligibility_score);
          return (
            <div
              key={t.nct_id || i}
              style={{
                background: "rgba(255,255,255,0.025)",
                border: "1px solid #1e3a5f",
                borderRadius: 10,
                padding: "0.8rem 1rem",
              }}
            >
              {/* Row 1: title + badge */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "0.8rem",
                  marginBottom: 5,
                }}
              >
                <div
                  style={{
                    fontSize: "0.82rem",
                    fontWeight: 600,
                    color: "#e2e8f0",
                    lineHeight: 1.4,
                    flex: 1,
                  }}
                >
                  {t.title}
                </div>
                <span
                  style={{
                    fontSize: "0.68rem",
                    padding: "2px 9px",
                    borderRadius: 20,
                    border: `1px solid ${b.color}`,
                    color: b.color,
                    background: b.bg,
                    whiteSpace: "nowrap",
                    flexShrink: 0,
                  }}
                >
                  {b.label}
                </span>
              </div>

              {/* Row 2: meta chips */}
              <div
                style={{
                  display: "flex",
                  gap: 5,
                  flexWrap: "wrap",
                  marginBottom: 6,
                }}
              >
                <span
                  style={{
                    fontSize: "0.67rem",
                    padding: "1px 7px",
                    borderRadius: 4,
                    border: `1px solid ${phaseColor(t.phase)}`,
                    color: phaseColor(t.phase),
                    background: "rgba(255,255,255,0.03)",
                  }}
                >
                  {t.phase}
                </span>
                <span
                  style={{
                    fontSize: "0.67rem",
                    padding: "1px 7px",
                    borderRadius: 4,
                    border: "1px solid #1e3a5f",
                    color: "#64748b",
                    background: "rgba(255,255,255,0.03)",
                  }}
                >
                  {t.study_status}
                </span>
                {t.biomarker_focus && (
                  <span
                    style={{
                      fontSize: "0.67rem",
                      padding: "1px 7px",
                      borderRadius: 4,
                      border: "1px solid rgba(245,158,11,0.3)",
                      color: "#f59e0b",
                      background: "rgba(245,158,11,0.05)",
                    }}
                  >
                    {t.biomarker_focus}
                  </span>
                )}
                {t.nct_id && (
                  <a
                    href={`https://clinicaltrials.gov/study/${t.nct_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      fontSize: "0.67rem",
                      padding: "1px 7px",
                      borderRadius: 4,
                      border: "1px solid #1e4a8f",
                      color: "#60a5fa",
                      background: "rgba(96,165,250,0.05)",
                      textDecoration: "none",
                    }}
                  >
                    {t.nct_id} ↗
                  </a>
                )}
              </div>

              {/* Brief summary */}
              {t.brief_summary && (
                <div
                  style={{
                    fontSize: "0.73rem",
                    color: "#64748b",
                    marginBottom: 6,
                    lineHeight: 1.5,
                  }}
                >
                  {t.brief_summary.slice(0, 160)}
                  {t.brief_summary.length > 160 ? "…" : ""}
                </div>
              )}

              {/* Eligibility flags */}
              {t.eligibility_flags && Object.keys(t.eligibility_flags).length > 0 && (
                <div
                  style={{
                    display: "flex",
                    gap: 4,
                    flexWrap: "wrap",
                  }}
                >
                  {Object.entries(t.eligibility_flags).map(([key, val]) => (
                    <span
                      key={key}
                      style={{
                        fontSize: "0.65rem",
                        padding: "1px 7px",
                        borderRadius: 4,
                        border: `1px solid ${flagColor(val)}`,
                        color: flagColor(val),
                        background: "rgba(255,255,255,0.02)",
                      }}
                    >
                      {key}: {val}
                    </span>
                  ))}
                </div>
              )}

              {/* Match score */}
              <div
                style={{
                  marginTop: 6,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <div
                  style={{
                    flex: 1,
                    height: 4,
                    background: "rgba(255,255,255,0.06)",
                    borderRadius: 2,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${(t.eligibility_score * 100).toFixed(0)}%`,
                      background: b.color,
                      borderRadius: 2,
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
                <span
                  style={{ fontSize: "0.65rem", color: b.color, minWidth: 30, textAlign: "right" }}
                >
                  {(t.eligibility_score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
