"use client";

/**
 * Plain-English patient summary tab.
 * Visual contrast: large readable serif font on a soft warm background,
 * deliberately different from the dark clinical UI — signals "this is for the patient".
 */

interface PatientSummaryTabProps {
  summary: string;
  diagnosis: string;
  stage: string;
  firstLineTreatment: string;
  immediateActions: string[];
}

export default function PatientSummaryTab({
  summary,
  diagnosis,
  stage,
  firstLineTreatment,
  immediateActions,
}: PatientSummaryTabProps) {
  return (
    <div
      style={{
        background: "rgba(255,251,240,0.04)",
        border: "1px solid rgba(251,243,219,0.12)",
        borderRadius: "12px",
        padding: "1.75rem 2rem",
        fontFamily: "Georgia, 'Times New Roman', serif",
        color: "rgba(240,232,210,0.95)",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          marginBottom: "1.5rem",
          paddingBottom: "1rem",
          borderBottom: "1px solid rgba(251,243,219,0.1)",
        }}
      >
        <span style={{ fontSize: "1.5rem" }}>👤</span>
        <div>
          <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "rgba(251,243,219,0.9)", lineHeight: 1.2 }}>
            Your Personal Health Summary
          </div>
          <div style={{ fontSize: "0.78rem", color: "rgba(240,232,210,0.5)", marginTop: "0.2rem", fontFamily: "sans-serif" }}>
            Written in plain English · 8th-grade reading level · Not for clinical use
          </div>
        </div>
      </div>

      {/* Main summary paragraph */}
      <p style={{ fontSize: "1rem", lineHeight: 1.85, marginBottom: "1.5rem", color: "rgba(240,232,210,0.9)" }}>
        {summary || "Your care team has completed a thorough review of your case. Please speak with your doctor to understand what these findings mean for you personally."}
      </p>

      {/* Key facts in plain language */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "1rem",
          marginBottom: "1.5rem",
        }}
      >
        <div
          style={{
            padding: "1rem",
            background: "rgba(251,243,219,0.04)",
            border: "1px solid rgba(251,243,219,0.1)",
            borderRadius: "10px",
          }}
        >
          <div style={{ fontSize: "0.7rem", color: "rgba(240,232,210,0.45)", fontFamily: "sans-serif", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.4rem" }}>
            What was found
          </div>
          <div style={{ fontSize: "0.9rem", lineHeight: 1.5 }}>{diagnosis}</div>
        </div>

        <div
          style={{
            padding: "1rem",
            background: "rgba(251,243,219,0.04)",
            border: "1px solid rgba(251,243,219,0.1)",
            borderRadius: "10px",
          }}
        >
          <div style={{ fontSize: "0.7rem", color: "rgba(240,232,210,0.45)", fontFamily: "sans-serif", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.4rem" }}>
            Stage
          </div>
          <div style={{ fontSize: "0.9rem", lineHeight: 1.5 }}>{stage}</div>
        </div>

        <div
          style={{
            padding: "1rem",
            background: "rgba(251,243,219,0.04)",
            border: "1px solid rgba(251,243,219,0.1)",
            borderRadius: "10px",
          }}
        >
          <div style={{ fontSize: "0.7rem", color: "rgba(240,232,210,0.45)", fontFamily: "sans-serif", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.4rem" }}>
            Recommended treatment
          </div>
          <div style={{ fontSize: "0.9rem", lineHeight: 1.5 }}>{firstLineTreatment}</div>
        </div>
      </div>

      {/* Next steps */}
      {immediateActions.length > 0 && (
        <div>
          <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "rgba(251,243,219,0.6)", marginBottom: "0.6rem" }}>
            Your next steps:
          </div>
          <ol style={{ margin: 0, paddingLeft: "1.25rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {immediateActions.map((action, i) => (
              <li key={i} style={{ fontSize: "0.88rem", lineHeight: 1.65, color: "rgba(240,232,210,0.85)" }}>
                {action}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Disclaimer */}
      <div
        style={{
          marginTop: "1.5rem",
          padding: "0.65rem 0.9rem",
          background: "rgba(239,68,68,0.05)",
          border: "1px solid rgba(239,68,68,0.15)",
          borderRadius: "8px",
          fontSize: "0.72rem",
          color: "rgba(252,165,165,0.7)",
          fontFamily: "sans-serif",
          lineHeight: 1.5,
        }}
      >
        ⚠️ This summary is generated by an AI research tool and is NOT a substitute for advice from your doctor. Always discuss treatment decisions with a qualified oncologist.
      </div>
    </div>
  );
}
