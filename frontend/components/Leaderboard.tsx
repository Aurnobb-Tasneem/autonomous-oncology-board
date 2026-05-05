"use client";

import { type AblationData, type AblationRow } from "@/lib/eval-data";

interface LeaderboardProps {
  data: AblationData;
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function ci(low: number, high: number) {
  return `[${pct(low)}, ${pct(high)}]`;
}

function MetricCell({
  mean,
  ciLow,
  ciHigh,
  isFull,
  delta,
}: {
  mean: number;
  ciLow: number;
  ciHigh: number;
  isFull: boolean;
  delta?: number;
}) {
  return (
    <td
      style={{
        padding: "0.65rem 0.75rem",
        textAlign: "center",
        verticalAlign: "middle",
      }}
    >
      <div
        style={{
          fontSize: "0.92rem",
          fontWeight: isFull ? 800 : 600,
          color: isFull ? "var(--teal-light)" : "var(--text-secondary)",
        }}
      >
        {pct(mean)}
      </div>
      <div style={{ fontSize: "0.62rem", color: "var(--text-muted)", marginTop: "0.1rem" }}>
        {ci(ciLow, ciHigh)}
      </div>
      {!isFull && delta !== undefined && (
        <div
          style={{
            fontSize: "0.65rem",
            color: delta < 0 ? "#f87171" : "#4ade80",
            fontWeight: 700,
            marginTop: "0.1rem",
          }}
        >
          {delta < 0 ? "▼" : "▲"} {pct(Math.abs(delta))}
        </div>
      )}
    </td>
  );
}

export default function Leaderboard({ data }: LeaderboardProps) {
  const headers = ["Model / Config", "TNM Acc.", "Biomarker F1", "Tx Alignment", "Schema Valid."];

  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "0.83rem",
        }}
      >
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th
                key={i}
                style={{
                  padding: "0.6rem 0.75rem",
                  textAlign: i === 0 ? "left" : "center",
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  borderBottom: "1px solid rgba(148,163,184,0.15)",
                  whiteSpace: "nowrap",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.ablation_table.map((row: AblationRow, idx) => (
            <tr
              key={idx}
              style={{
                background: row.is_full
                  ? "rgba(13,148,136,0.08)"
                  : idx % 2 === 0
                  ? "rgba(255,255,255,0.015)"
                  : "transparent",
                borderBottom: "1px solid rgba(148,163,184,0.07)",
              }}
            >
              <td style={{ padding: "0.65rem 0.75rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  {row.is_full && (
                    <span
                      style={{
                        padding: "0.15rem 0.45rem",
                        borderRadius: "20px",
                        fontSize: "0.62rem",
                        fontWeight: 700,
                        background: "rgba(13,148,136,0.2)",
                        border: "1px solid rgba(13,148,136,0.4)",
                        color: "var(--teal-light)",
                        whiteSpace: "nowrap",
                      }}
                    >
                      AOB
                    </span>
                  )}
                  <span
                    style={{
                      fontWeight: row.is_full ? 700 : 400,
                      color: row.is_full ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    {row.config}
                  </span>
                </div>
              </td>
              <MetricCell
                mean={row.tnm_accuracy.mean}
                ciLow={row.tnm_accuracy.ci_low}
                ciHigh={row.tnm_accuracy.ci_high}
                isFull={row.is_full}
                delta={row.is_full ? undefined : row.delta_tnm}
              />
              <MetricCell
                mean={row.biomarker_f1.mean}
                ciLow={row.biomarker_f1.ci_low}
                ciHigh={row.biomarker_f1.ci_high}
                isFull={row.is_full}
              />
              <MetricCell
                mean={row.tx_alignment.mean}
                ciLow={row.tx_alignment.ci_low}
                ciHigh={row.tx_alignment.ci_high}
                isFull={row.is_full}
              />
              <MetricCell
                mean={row.schema_validity.mean}
                ciLow={row.schema_validity.ci_low}
                ciHigh={row.schema_validity.ci_high}
                isFull={row.is_full}
              />
            </tr>
          ))}
        </tbody>
      </table>

      <div style={{ marginTop: "0.75rem", fontSize: "0.7rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
        95% bootstrap CIs (N={data.n_cases} cases, 1 000 resamples). TNM Acc. = exact TNM stage match.
        Biomarker F1 = macro-averaged F1 over EGFR/ALK/ROS1/KRAS. Tx Alignment = NCCN guideline alignment.
        Schema Valid. = structured JSON output conformance.
      </div>
    </div>
  );
}
