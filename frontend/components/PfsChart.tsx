"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface PfsPoint { month: number; pfs: number; }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div
        style={{
          background: "rgba(10,22,40,0.95)",
          border: "1px solid var(--teal-border)",
          borderRadius: "8px",
          padding: "0.6rem 0.8rem",
          fontSize: "0.8rem",
        }}
      >
        <div style={{ color: "var(--text-muted)", marginBottom: "0.2rem" }}>Month {label}</div>
        <div style={{ color: "var(--teal-light)", fontWeight: 700 }}>
          PFS: {(payload[0].value * 100).toFixed(1)}%
        </div>
      </div>
    );
  }
  return null;
};

export default function PfsChart({
  data,
  pfs12mo,
}: {
  data: PfsPoint[];
  pfs12mo?: number;
}) {
  if (!data || data.length === 0) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: "0.85rem", padding: "1rem 0" }}>
        Digital twin simulation data not available.
      </div>
    );
  }

  const formatted = data.map((d) => ({ ...d, pfs: d.pfs }));

  return (
    <div>
      {pfs12mo !== undefined && (
        <div style={{ marginBottom: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>Predicted 12-month PFS:</span>
          <span
            style={{
              fontSize: "1.1rem",
              fontWeight: 700,
              color: "var(--teal-light)",
              textShadow: "0 0 12px rgba(13,148,136,0.5)",
            }}
          >
            {(pfs12mo * 100).toFixed(1)}%
          </span>
        </div>
      )}
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={formatted} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="month"
            tick={{ fill: "#64748b", fontSize: 11 }}
            label={{ value: "Months", position: "insideBottom", fill: "#64748b", fontSize: 11, dy: 8 }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
          />
          <YAxis
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            domain={[0, 1]}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={0.5}
            stroke="rgba(245,158,11,0.35)"
            strokeDasharray="4 4"
            label={{ value: "50%", fill: "#f59e0b", fontSize: 10, position: "right" }}
          />
          <Line
            type="monotone"
            dataKey="pfs"
            stroke="#0d9488"
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 5, fill: "#14b8a6", stroke: "#0d9488", strokeWidth: 2 }}
            style={{ filter: "drop-shadow(0 0 4px rgba(13,148,136,0.5))" }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", textAlign: "center", marginTop: "0.25rem" }}>
        Digital Twin — ODE-based 12-month progression-free survival simulation
      </p>
    </div>
  );
}
