"use client";

interface Differential {
  diagnosis: string;
  probability: number;
  supporting_features: string[];
}

interface DifferentialResult {
  differentials: Differential[];
  rule_out_tests: string[];
  primary_confidence: string;
  source: string;
}

interface Props {
  data: DifferentialResult | null;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high:     "#22d3ee",
  moderate: "#f59e0b",
  low:      "#f87171",
};

const BAR_COLORS = [
  "linear-gradient(90deg, #22d3ee, #0e7490)",
  "linear-gradient(90deg, #6366f1, #4338ca)",
  "linear-gradient(90deg, #8b5cf6, #6d28d9)",
];

export default function DifferentialPanel({ data }: Props) {
  if (!data || !data.differentials?.length) return null;

  const confColor =
    CONFIDENCE_COLORS[data.primary_confidence?.toLowerCase()] ?? "#94a3b8";

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
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "0.9rem",
        }}
      >
        <h3 style={{ margin: 0, color: "#e2e8f0", fontSize: "0.95rem", fontWeight: 700 }}>
          🔬 Differential Diagnosis
        </h3>
        <span
          style={{
            fontSize: "0.7rem",
            padding: "2px 10px",
            borderRadius: 20,
            border: `1px solid ${confColor}`,
            color: confColor,
            background: "rgba(255,255,255,0.04)",
          }}
        >
          {data.primary_confidence} confidence
        </span>
      </div>

      {/* Stacked probability bars */}
      <div style={{ marginBottom: "1rem" }}>
        <div
          style={{
            height: 18,
            borderRadius: 9,
            overflow: "hidden",
            display: "flex",
            gap: 1,
          }}
        >
          {data.differentials.map((d, i) => (
            <div
              key={i}
              title={`${d.diagnosis}: ${(d.probability * 100).toFixed(0)}%`}
              style={{
                flex:       d.probability,
                background: BAR_COLORS[i] ?? "#334155",
                transition: "flex 0.6s ease",
              }}
            />
          ))}
        </div>
        <div
          style={{
            display: "flex",
            gap: "1rem",
            marginTop: "0.5rem",
            flexWrap: "wrap",
          }}
        >
          {data.differentials.map((d, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 5,
                fontSize: "0.7rem",
                color: "#94a3b8",
              }}
            >
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 2,
                  background: BAR_COLORS[i] ?? "#334155",
                  flexShrink: 0,
                }}
              />
              <span style={{ color: i === 0 ? "#e2e8f0" : "#94a3b8" }}>
                {d.diagnosis} ({(d.probability * 100).toFixed(0)}%)
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Individual differentials */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem" }}>
        {data.differentials.map((d, i) => (
          <div
            key={i}
            style={{
              background: i === 0 ? "rgba(14,116,144,0.08)" : "rgba(255,255,255,0.02)",
              border: `1px solid ${i === 0 ? "#0e7490" : "#1e3a5f"}`,
              borderRadius: 8,
              padding: "0.7rem 0.9rem",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 4,
              }}
            >
              <span
                style={{
                  fontSize: "0.85rem",
                  fontWeight: i === 0 ? 700 : 500,
                  color: i === 0 ? "#22d3ee" : "#94a3b8",
                }}
              >
                {i === 0 ? "① " : i === 1 ? "② " : "③ "}
                {d.diagnosis}
              </span>
              <span
                style={{
                  fontSize: "0.9rem",
                  fontWeight: 700,
                  color: BAR_COLORS[i]
                    ? ["#22d3ee", "#818cf8", "#a78bfa"][i]
                    : "#94a3b8",
                }}
              >
                {(d.probability * 100).toFixed(0)}%
              </span>
            </div>
            {d.supporting_features?.length > 0 && (
              <div
                style={{
                  display: "flex",
                  gap: 4,
                  flexWrap: "wrap",
                }}
              >
                {d.supporting_features.slice(0, 4).map((f, j) => (
                  <span
                    key={j}
                    style={{
                      fontSize: "0.67rem",
                      color: "#64748b",
                      background: "rgba(255,255,255,0.04)",
                      border: "1px solid #1e3a5f",
                      borderRadius: 4,
                      padding: "1px 6px",
                    }}
                  >
                    {f}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Rule-out tests */}
      {data.rule_out_tests?.length > 0 && (
        <div style={{ marginTop: "0.9rem" }}>
          <div
            style={{
              fontSize: "0.72rem",
              color: "#64748b",
              fontWeight: 600,
              marginBottom: 5,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Rule-Out Tests
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {data.rule_out_tests.map((t, i) => (
              <span
                key={i}
                style={{
                  fontSize: "0.7rem",
                  color: "#f59e0b",
                  background: "rgba(245,158,11,0.08)",
                  border: "1px solid rgba(245,158,11,0.25)",
                  borderRadius: 4,
                  padding: "2px 8px",
                }}
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      <div
        style={{
          marginTop: "0.7rem",
          fontSize: "0.65rem",
          color: "#334155",
          textAlign: "right",
        }}
      >
        source: {data.source}
      </div>
    </div>
  );
}
