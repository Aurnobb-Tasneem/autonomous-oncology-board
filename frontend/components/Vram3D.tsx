"use client";

/**
 * Pure CSS 3D rotating cube showing VRAM shares per model.
 * No three.js, no WebGL — just CSS transform-style: preserve-3d.
 * Falls back gracefully if the browser doesn't support 3D transforms.
 */

const TOTAL_GB = 192;

const FACES = [
  { label: "Llama 3.3\n70B", gb: 70, color: "#0891b2", accentColor: "#38bdf8" },
  { label: "KV Cache\n30 GB", gb: 30, color: "#7c3aed", accentColor: "#a78bfa" },
  { label: "Qwen-VL\n15 GB", gb: 15, color: "#22c55e", accentColor: "#4ade80" },
  { label: "LoRA ×3\n16 GB", gb: 16, color: "#38bdf8", accentColor: "#7dd3fc" },
  { label: "GigaPath\n3 GB", gb: 3, color: "#0d9488", accentColor: "#2dd4bf" },
  { label: "Qdrant\n+Misc 9 GB", gb: 9, color: "#64748b", accentColor: "#94a3b8" },
];

function CubeFace({
  transform,
  face,
}: {
  transform: string;
  face: (typeof FACES)[0];
}) {
  const pct = Math.round((face.gb / TOTAL_GB) * 100);
  return (
    <div
      style={{
        position: "absolute",
        width: "160px",
        height: "160px",
        border: `1px solid ${face.accentColor}60`,
        background: `linear-gradient(135deg, ${face.color}25, ${face.color}10)`,
        backdropFilter: "blur(4px)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "0.3rem",
        transform,
        backfaceVisibility: "visible",
      }}
    >
      <div style={{ fontSize: "1.4rem", fontWeight: 800, color: face.accentColor, lineHeight: 1 }}>
        {face.gb} GB
      </div>
      <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.7)", textAlign: "center", lineHeight: 1.4, whiteSpace: "pre-line" }}>
        {face.label}
      </div>
      <div style={{ fontSize: "0.6rem", color: `${face.accentColor}90` }}>{pct}% of 192 GB</div>
    </div>
  );
}

export default function Vram3D() {
  const s = 160; // cube side in px
  const half = s / 2;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.5rem" }}>
      <div
        style={{
          perspective: "600px",
          width: `${s}px`,
          height: `${s}px`,
        }}
      >
        <div
          style={{
            width: `${s}px`,
            height: `${s}px`,
            position: "relative",
            transformStyle: "preserve-3d",
            animation: "cube-spin 14s linear infinite",
          }}
        >
          <CubeFace transform={`translateZ(${half}px)`} face={FACES[0]} />
          <CubeFace transform={`rotateY(180deg) translateZ(${half}px)`} face={FACES[1]} />
          <CubeFace transform={`rotateY(90deg) translateZ(${half}px)`} face={FACES[2]} />
          <CubeFace transform={`rotateY(-90deg) translateZ(${half}px)`} face={FACES[3]} />
          <CubeFace transform={`rotateX(90deg) translateZ(${half}px)`} face={FACES[4]} />
          <CubeFace transform={`rotateX(-90deg) translateZ(${half}px)`} face={FACES[5]} />
        </div>
      </div>

      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textAlign: "center" }}>
        AMD MI300X · 192 GB HBM3 · All models resident simultaneously
      </div>

      <style>{`
        @keyframes cube-spin {
          0% { transform: rotateX(-15deg) rotateY(0deg); }
          100% { transform: rotateX(-15deg) rotateY(360deg); }
        }
      `}</style>
    </div>
  );
}
