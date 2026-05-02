"use client";

export default function ConfidenceRing({
  value,
  size = 100,
}: {
  value: number;
  size?: number;
}) {
  const pct = Math.min(Math.max(value, 0), 1);
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - pct);
  const color = pct >= 0.75 ? "#14b8a6" : pct >= 0.5 ? "#f59e0b" : "#ef4444";

  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={8}
        />
        {/* Progress arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{
            transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1), stroke 0.5s ease",
            filter: `drop-shadow(0 0 6px ${color})`,
          }}
        />
      </svg>
      {/* Center label */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: size * 0.2, fontWeight: 700, color, lineHeight: 1 }}>
          {Math.round(pct * 100)}%
        </span>
        <span style={{ fontSize: size * 0.12, color: "var(--text-muted)", lineHeight: 1.2 }}>
          confidence
        </span>
      </div>
    </div>
  );
}
