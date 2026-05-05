"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { type CalibrationData } from "@/lib/eval-data";

interface CalibrationCurveProps {
  data: CalibrationData;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      style={{
        background: "rgba(5,15,30,0.97)",
        border: "1px solid #0e7490",
        borderRadius: 8,
        padding: "0.6rem 0.9rem",
        fontSize: "0.78rem",
      }}
    >
      <div style={{ color: "#94a3b8", marginBottom: "0.3rem" }}>
        Conf bin: {(d.bin_mid * 100).toFixed(0)}%
      </div>
      <div style={{ color: "#22d3ee", fontWeight: 700 }}>
        Observed: {(d.fraction_positive * 100).toFixed(1)}%
      </div>
    </div>
  );
};

function StatBadge({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "0.5rem 0.75rem",
        borderRadius: "8px",
        background: `${color}10`,
        border: `1px solid ${color}30`,
        minWidth: "80px",
      }}
    >
      <span style={{ fontSize: "1rem", fontWeight: 800, color }}>{value}</span>
      <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.15rem" }}>{label}</span>
    </div>
  );
}

export default function CalibrationCurve({ data }: CalibrationCurveProps) {
  const gigapathPoints = data.gigapath.reliability_curve.map((p) => ({
    bin_mid: p.bin_mid,
    fraction_positive: p.fraction_positive,
  }));
  const boardPoints = data.board_consensus.reliability_curve.map((p) => ({
    bin_mid: p.bin_mid,
    fraction_positive: p.fraction_positive,
  }));

  // Perfect calibration reference line data
  const perfectLine = Array.from({ length: 11 }, (_, i) => ({
    x: i * 0.1,
    y: i * 0.1,
  }));

  return (
    <div>
      <div style={{ marginBottom: "1rem" }}>
        <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.3rem" }}>
          Calibration Reliability Curves
        </div>
        <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
          Predicted confidence vs. observed accuracy. Diagonal = perfect calibration.
        </div>
      </div>

      {/* Stat badges */}
      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: "0.65rem", color: "#0d9488", fontWeight: 700, marginBottom: "0.4rem" }}>GigaPath</div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <StatBadge label="ECE" value={(data.gigapath.ece * 100).toFixed(1) + "%"} color="#0d9488" />
            <StatBadge label="MCE" value={(data.gigapath.mce * 100).toFixed(1) + "%"} color="#0d9488" />
            <StatBadge label="Brier" value={data.gigapath.brier.toFixed(3)} color="#0d9488" />
          </div>
        </div>
        <div>
          <div style={{ fontSize: "0.65rem", color: "#0891b2", fontWeight: 700, marginBottom: "0.4rem" }}>Board Consensus</div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <StatBadge label="ECE" value={(data.board_consensus.ece * 100).toFixed(1) + "%"} color="#0891b2" />
            <StatBadge label="MCE" value={(data.board_consensus.mce * 100).toFixed(1) + "%"} color="#0891b2" />
            <StatBadge label="Brier" value={data.board_consensus.brier.toFixed(3)} color="#0891b2" />
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="bin_mid"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#475569", fontSize: 10 }}
            tickLine={false}
            name="Predicted Confidence"
          />
          <YAxis
            dataKey="fraction_positive"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#475569", fontSize: 10 }}
            tickLine={false}
            name="Observed Accuracy"
          />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3" }} />
          {/* Perfect calibration diagonal */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke="#475569"
            strokeDasharray="5 3"
            strokeWidth={1}
          />
          <Scatter
            name="GigaPath"
            data={gigapathPoints}
            fill="#0d9488"
            opacity={0.9}
            line={{ stroke: "#0d9488", strokeWidth: 2 }}
          />
          <Scatter
            name="Board Consensus"
            data={boardPoints}
            fill="#0891b2"
            opacity={0.9}
            line={{ stroke: "#0891b2", strokeWidth: 2 }}
          />
          <Legend
            wrapperStyle={{ fontSize: "0.72rem", color: "#94a3b8" }}
          />
        </ScatterChart>
      </ResponsiveContainer>

      <div style={{ marginTop: "0.5rem", fontSize: "0.68rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
        Lower ECE = better calibration. Board consensus ECE ({(data.board_consensus.ece * 100).toFixed(1)}%) outperforms GigaPath alone ({(data.gigapath.ece * 100).toFixed(1)}%),
        showing the deliberation loop improves confidence calibration.
      </div>
    </div>
  );
}
