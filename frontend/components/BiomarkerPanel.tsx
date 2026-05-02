"use client";

const BIOMARKER_LABELS: Record<string, string> = {
  nuclear_pleomorphism: "Nuclear Pleomorphism",
  mitotic_index: "Mitotic Index",
  gland_formation: "Gland Formation",
  necrosis_extent: "Necrosis Extent",
  immune_infiltration: "Immune Infiltration",
  stroma_density: "Stroma Density",
  cell_uniformity: "Cell Uniformity",
  architecture_score: "Architecture Score",
};

export default function BiomarkerPanel({
  biomarkers,
}: {
  biomarkers: Record<string, { score: number; level: string }>;
}) {
  const entries = Object.entries(biomarkers);
  if (entries.length === 0) {
    return <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>No biomarker data</div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem" }}>
      {entries.map(([key, val]) => {
        const label = BIOMARKER_LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        const pct = Math.round(val.score * 100);
        const level = val.level?.toLowerCase() ?? "low";
        const barClass =
          level === "high"
            ? "biomarker-high"
            : level === "moderate" || level === "medium"
            ? "biomarker-medium"
            : "biomarker-low";
        const levelColor =
          level === "high"
            ? "var(--teal-light)"
            : level === "moderate" || level === "medium"
            ? "var(--warning)"
            : "#3b82f6";

        return (
          <div key={key}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                {label}
              </span>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{pct}%</span>
                <span
                  style={{
                    fontSize: "0.7rem",
                    color: levelColor,
                    border: `1px solid ${levelColor}`,
                    borderRadius: "10px",
                    padding: "0.05rem 0.45rem",
                    fontWeight: 600,
                    opacity: 0.9,
                  }}
                >
                  {val.level}
                </span>
              </div>
            </div>
            <div
              style={{
                height: "8px",
                background: "rgba(255,255,255,0.05)",
                borderRadius: "4px",
                overflow: "hidden",
              }}
            >
              <div
                className={`bar-fill ${barClass}`}
                style={{ "--fill-width": `${pct}%` } as React.CSSProperties}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
