"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { type AblationData } from "@/lib/eval-data";

interface AblationChartProps {
  data: AblationData;
  metric?: "tnm_accuracy" | "biomarker_f1" | "tx_alignment";
}

const METRIC_LABELS = {
  tnm_accuracy: "TNM Accuracy",
  biomarker_f1: "Biomarker F1",
  tx_alignment: "Treatment Alignment",
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      style={{
        background: "rgba(5,15,30,0.97)",
        border: "1px solid #0e7490",
        borderRadius: 8,
        padding: "0.65rem 0.9rem",
        fontSize: "0.78rem",
        minWidth: "160px",
      }}
    >
      <div style={{ color: "#94a3b8", marginBottom: "0.4rem", fontWeight: 600 }}>
        {d.config}
      </div>
      <div style={{ color: d.delta < 0 ? "#f87171" : "#22d3ee", fontWeight: 700, marginBottom: "0.2rem" }}>
        Δ = {(d.delta * 100).toFixed(1)}%
      </div>
      <div style={{ color: "#64748b" }}>
        Score: {(d.score * 100).toFixed(1)}%
      </div>
    </div>
  );
};

export default function AblationChart({
  data,
  metric = "tnm_accuracy",
}: AblationChartProps) {
  const ablationRows = data.ablation_table.filter((r) => !r.is_full);
  const fullRow = data.ablation_table.find((r) => r.is_full);
  const fullScore = fullRow ? fullRow[metric].mean : 0;

  const chartData = ablationRows.map((r) => ({
    config: r.config_short,
    score: r[metric].mean,
    delta: r[metric].mean - fullScore,
  }));

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "0.75rem",
          flexWrap: "wrap",
          gap: "0.5rem",
        }}
      >
        <div>
          <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)" }}>
            Ablation Study — {METRIC_LABELS[metric]}
          </div>
          <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "0.15rem" }}>
            Δ vs. full AOB pipeline ({(fullScore * 100).toFixed(1)}%). Negative = component is contributing.
          </div>
        </div>
        <div
          style={{
            padding: "0.25rem 0.7rem",
            borderRadius: "20px",
            background: "rgba(13,148,136,0.12)",
            border: "1px solid rgba(13,148,136,0.35)",
            fontSize: "0.72rem",
            fontWeight: 700,
            color: "var(--teal-light)",
          }}
        >
          Full: {(fullScore * 100).toFixed(1)}%
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 60, left: 4, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
          <XAxis
            type="number"
            domain={[-0.18, 0.01]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#475569", fontSize: 10 }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="config"
            width={90}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
          <ReferenceLine x={0} stroke="#0e7490" strokeWidth={1.5} strokeDasharray="4 2" />
          <Bar dataKey="delta" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.delta < -0.07 ? "#ef4444" : entry.delta < -0.03 ? "#f59e0b" : "#22d3ee"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
