"use client";

import { useEffect, useRef, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface VramPoint {
  ts: number;
  used_gb: number;
  total_gb: number;
  pct: number;
}

interface Props {
  pollIntervalMs?: number;
  windowSeconds?: number;
  showH100Line?: boolean;
}

const H100_LIMIT_GB   = 80;
const MI300X_TOTAL_GB = 192;

function formatElapsed(ts: number, oldest: number): string {
  const s = Math.round(ts - oldest);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${s % 60}s`;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const used = payload.find((p: any) => p.dataKey === "used_gb");
  const pct  = payload.find((p: any) => p.dataKey === "pct");
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
      <div style={{ color: "#94a3b8", marginBottom: 4 }}>T+{label}</div>
      {used && (
        <div style={{ color: "#22d3ee", fontWeight: 700 }}>
          Used: {used.value.toFixed(1)} GB
        </div>
      )}
      {pct && (
        <div style={{ color: "#64748b" }}>{pct.value.toFixed(1)}% of 192 GB</div>
      )}
      {used && used.value > H100_LIMIT_GB && (
        <div style={{ color: "#f87171", marginTop: 4 }}>
          ⚠️ Would OOM on H100 (80 GB)
        </div>
      )}
    </div>
  );
};

export default function VramTimeSeries({
  pollIntervalMs = 2000,
  windowSeconds  = 300,
  showH100Line   = true,
}: Props) {
  const [points, setPoints]     = useState<VramPoint[]>([]);
  const [currentGb, setCurrentGb] = useState(0);
  const [oomIfH100, setOomIfH100] = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchVram = async () => {
    try {
      const res = await fetch(`/api/vram/history?seconds=${windowSeconds}`);
      if (!res.ok) return;
      const data = await res.json();
      setPoints(data.points ?? []);
      setCurrentGb(data.current_gb ?? 0);
      setOomIfH100(data.oom_if_h100 ?? false);
      setError(null);
    } catch (e: any) {
      setError(e.message ?? "fetch error");
    }
  };

  useEffect(() => {
    fetchVram();
    timerRef.current = setInterval(fetchVram, pollIntervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [pollIntervalMs, windowSeconds]);

  const oldest   = points[0]?.ts ?? Date.now() / 1000;
  const chartData = points.map((p) => ({
    label:   formatElapsed(p.ts, oldest),
    used_gb: p.used_gb,
    pct:     p.pct,
  }));

  const usedPct    = (currentGb / MI300X_TOTAL_GB) * 100;
  const h100UsedPct = Math.min(100, (currentGb / H100_LIMIT_GB) * 100);

  return (
    <div
      style={{
        background: "linear-gradient(135deg, rgba(10,22,40,0.9), rgba(5,10,25,0.95))",
        border: "1px solid #0e7490",
        borderRadius: 12,
        padding: "1.2rem 1.4rem",
        fontFamily: "var(--font-sans, sans-serif)",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1rem",
        }}
      >
        <div>
          <h3
            style={{ margin: 0, color: "#e2e8f0", fontSize: "1rem", fontWeight: 700 }}
          >
            🖥️ Live VRAM — AMD MI300X
          </h3>
          <div style={{ color: "#64748b", fontSize: "0.72rem", marginTop: 2 }}>
            192 GB HBM3 · polled every 2s · last {windowSeconds}s window
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            style={{
              fontSize: "1.6rem",
              fontWeight: 800,
              color: oomIfH100 ? "#f87171" : "#22d3ee",
              lineHeight: 1.1,
            }}
          >
            {currentGb.toFixed(1)} GB
          </div>
          <div style={{ color: "#64748b", fontSize: "0.72rem" }}>
            {usedPct.toFixed(1)}% utilised
          </div>
        </div>
      </div>

      {/* MI300X fill bar */}
      <div style={{ marginBottom: "0.6rem" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.7rem",
            color: "#64748b",
            marginBottom: 3,
          }}
        >
          <span>MI300X 192 GB</span>
          <span>{currentGb.toFixed(1)} / 192 GB</span>
        </div>
        <div
          style={{
            height: 12,
            background: "rgba(255,255,255,0.06)",
            borderRadius: 6,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${usedPct}%`,
              background:
                usedPct < 50
                  ? "linear-gradient(90deg, #0e7490, #22d3ee)"
                  : usedPct < 80
                  ? "linear-gradient(90deg, #0e7490, #f59e0b)"
                  : "linear-gradient(90deg, #991b1b, #ef4444)",
              borderRadius: 6,
              transition: "width 0.4s ease",
            }}
          />
        </div>
      </div>

      {/* H100 comparison bar */}
      {showH100Line && (
        <div style={{ marginBottom: "1rem" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.7rem",
              color: "#64748b",
              marginBottom: 3,
            }}
          >
            <span>H100 80 GB (comparison)</span>
            <span
              style={{ color: currentGb > H100_LIMIT_GB ? "#f87171" : "#64748b" }}
            >
              {currentGb > H100_LIMIT_GB
                ? `OOM — ${(currentGb - H100_LIMIT_GB).toFixed(1)} GB over limit`
                : `${H100_LIMIT_GB - currentGb < 0 ? 0 : (H100_LIMIT_GB - currentGb).toFixed(1)} GB headroom`}
            </span>
          </div>
          <div
            style={{
              height: 12,
              background: "rgba(255,255,255,0.06)",
              borderRadius: 6,
              overflow: "hidden",
              position: "relative",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${Math.min(100, h100UsedPct)}%`,
                background:
                  currentGb <= H100_LIMIT_GB
                    ? "linear-gradient(90deg, #374151, #6b7280)"
                    : "linear-gradient(90deg, #7f1d1d, #ef4444)",
                borderRadius: 6,
                transition: "width 0.4s ease",
              }}
            />
            {currentGb <= H100_LIMIT_GB && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  right: 0,
                  height: "100%",
                  width: 2,
                  background: "#9ca3af",
                  opacity: 0.5,
                }}
              />
            )}
          </div>
        </div>
      )}

      {/* Timeseries chart */}
      {chartData.length > 1 ? (
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart
            data={chartData}
            margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
          >
            <defs>
              <linearGradient id="vramGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#22d3ee" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="label"
              tick={{ fill: "#475569", fontSize: 10 }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, MI300X_TOTAL_GB]}
              tick={{ fill: "#475569", fontSize: 10 }}
              tickLine={false}
              unit=" GB"
            />
            <Tooltip content={<CustomTooltip />} />
            {showH100Line && (
              <ReferenceLine
                y={H100_LIMIT_GB}
                stroke="#ef4444"
                strokeDasharray="6 3"
                strokeWidth={1.5}
                label={{
                  value: "H100 limit 80 GB",
                  fill: "#ef4444",
                  fontSize: 10,
                  position: "insideTopRight",
                }}
              />
            )}
            <ReferenceLine
              y={MI300X_TOTAL_GB}
              stroke="#22d3ee"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{
                value: "MI300X 192 GB",
                fill: "#22d3ee",
                fontSize: 10,
                position: "insideBottomRight",
              }}
            />
            <Area
              type="monotone"
              dataKey="used_gb"
              stroke="#22d3ee"
              strokeWidth={2}
              fill="url(#vramGrad)"
              dot={false}
              activeDot={{ r: 3, fill: "#22d3ee" }}
              name="VRAM Used (GB)"
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div
          style={{
            height: 120,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#475569",
            fontSize: "0.8rem",
          }}
        >
          {error ? `Error: ${error}` : "Collecting VRAM samples…"}
        </div>
      )}

      {/* Status badge */}
      <div
        style={{
          marginTop: "0.8rem",
          display: "flex",
          gap: "0.6rem",
          flexWrap: "wrap",
          fontSize: "0.7rem",
        }}
      >
        <span
          style={{
            background: "rgba(14,116,144,0.2)",
            border: "1px solid #0e7490",
            color: "#22d3ee",
            borderRadius: 20,
            padding: "2px 10px",
          }}
        >
          AMD MI300X
        </span>
        <span
          style={{
            background: "rgba(14,116,144,0.2)",
            border: "1px solid #0e7490",
            color: "#22d3ee",
            borderRadius: 20,
            padding: "2px 10px",
          }}
        >
          192 GB HBM3
        </span>
        <span
          style={{
            background: "rgba(14,116,144,0.2)",
            border: "1px solid #0e7490",
            color: "#22d3ee",
            borderRadius: 20,
            padding: "2px 10px",
          }}
        >
          ROCm
        </span>
        {currentGb > H100_LIMIT_GB && (
          <span
            style={{
              background: "rgba(153,27,27,0.3)",
              border: "1px solid #ef4444",
              color: "#f87171",
              borderRadius: 20,
              padding: "2px 10px",
              fontWeight: 700,
            }}
          >
            ⚠️ H100 would OOM
          </span>
        )}
      </div>
    </div>
  );
}
