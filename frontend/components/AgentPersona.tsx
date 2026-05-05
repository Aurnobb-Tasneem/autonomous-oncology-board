"use client";

export type PersonaId = "pathologist" | "qwen_vl" | "researcher" | "oncologist";
export type PersonaState = "idle" | "active" | "speaking";

interface AgentPersonaProps {
  id: PersonaId;
  state?: PersonaState;
  size?: number;
  showLabel?: boolean;
}

const PERSONAS: Record<
  PersonaId,
  { label: string; color: string; accentColor: string; bgColor: string }
> = {
  pathologist: {
    label: "GigaPath Pathologist",
    color: "#0d9488",
    accentColor: "#2dd4bf",
    bgColor: "rgba(13,148,136,0.12)",
  },
  qwen_vl: {
    label: "Qwen-VL Second Opinion",
    color: "#22c55e",
    accentColor: "#4ade80",
    bgColor: "rgba(34,197,94,0.12)",
  },
  researcher: {
    label: "Researcher",
    color: "#7c3aed",
    accentColor: "#a78bfa",
    bgColor: "rgba(124,58,237,0.12)",
  },
  oncologist: {
    label: "Lead Oncologist",
    color: "#0891b2",
    accentColor: "#38bdf8",
    bgColor: "rgba(8,145,178,0.12)",
  },
};

function PathologistSVG({ color, accent }: { color: string; accent: string }) {
  return (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Microscope body */}
      <rect x="24" y="38" width="16" height="18" rx="3" fill={color} opacity="0.8" />
      {/* Stage */}
      <rect x="18" y="46" width="28" height="4" rx="2" fill={accent} opacity="0.7" />
      {/* Arm */}
      <rect x="30" y="18" width="4" height="24" rx="2" fill={color} />
      {/* Eyepiece */}
      <rect x="24" y="14" width="12" height="7" rx="3" fill={color} />
      <rect x="28" y="8" width="8" height="8" rx="4" fill={accent} />
      {/* Lens */}
      <circle cx="32" cy="42" r="5" fill={accent} opacity="0.9" />
      <circle cx="32" cy="42" r="3" fill={color} />
      {/* Sample grid */}
      <line x1="24" y1="48" x2="40" y2="48" stroke={accent} strokeWidth="0.8" opacity="0.5" />
      <line x1="32" y1="44" x2="32" y2="52" stroke={accent} strokeWidth="0.8" opacity="0.5" />
    </svg>
  );
}

function QwenVLSVG({ color, accent }: { color: string; accent: string }) {
  return (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Brain outline */}
      <path
        d="M32 12 C22 12 16 18 16 26 C16 30 18 34 22 36 L22 46 C22 48 24 50 26 50 L38 50 C40 50 42 48 42 46 L42 36 C46 34 48 30 48 26 C48 18 42 12 32 12Z"
        fill={color}
        opacity="0.7"
      />
      {/* Brain split line */}
      <line x1="32" y1="14" x2="32" y2="46" stroke={accent} strokeWidth="1.5" opacity="0.6" />
      {/* Neural connections */}
      <circle cx="24" cy="22" r="2.5" fill={accent} />
      <circle cx="40" cy="22" r="2.5" fill={accent} />
      <circle cx="20" cy="30" r="2" fill={accent} opacity="0.8" />
      <circle cx="44" cy="30" r="2" fill={accent} opacity="0.8" />
      <circle cx="28" cy="35" r="2" fill={accent} opacity="0.8" />
      <circle cx="36" cy="35" r="2" fill={accent} opacity="0.8" />
      <line x1="24" y1="22" x2="40" y2="22" stroke={accent} strokeWidth="1" opacity="0.4" />
      <line x1="24" y1="22" x2="28" y2="35" stroke={accent} strokeWidth="0.8" opacity="0.35" />
      <line x1="40" y1="22" x2="36" y2="35" stroke={accent} strokeWidth="0.8" opacity="0.35" />
      {/* VLM eye/camera */}
      <circle cx="32" cy="27" r="4" fill={accent} opacity="0.9" />
      <circle cx="32" cy="27" r="2" fill={color} />
    </svg>
  );
}

function ResearcherSVG({ color, accent }: { color: string; accent: string }) {
  return (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Book stack */}
      <rect x="16" y="44" width="32" height="8" rx="2" fill={color} opacity="0.9" />
      <rect x="18" y="35" width="28" height="10" rx="2" fill={color} opacity="0.75" />
      <rect x="20" y="27" width="24" height="9" rx="2" fill={color} opacity="0.6" />
      {/* Book spines */}
      <rect x="16" y="44" width="4" height="8" rx="1" fill={accent} />
      <rect x="18" y="35" width="4" height="10" rx="1" fill={accent} opacity="0.8" />
      <rect x="20" y="27" width="4" height="9" rx="1" fill={accent} opacity="0.7" />
      {/* Citation dots */}
      <circle cx="32" cy="20" r="4" fill={accent} />
      <circle cx="22" cy="14" r="2.5" fill={accent} opacity="0.7" />
      <circle cx="42" cy="14" r="2.5" fill={accent} opacity="0.7" />
      <line x1="22" y1="14" x2="32" y2="20" stroke={accent} strokeWidth="1" opacity="0.5" />
      <line x1="42" y1="14" x2="32" y2="20" stroke={accent} strokeWidth="1" opacity="0.5" />
    </svg>
  );
}

function OncologistSVG({ color, accent }: { color: string; accent: string }) {
  return (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Stethoscope tube */}
      <path
        d="M20 20 C20 20 14 22 14 30 C14 38 20 42 28 44"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
        fill="none"
        opacity="0.9"
      />
      {/* Chest piece */}
      <circle cx="32" cy="46" r="7" fill={color} opacity="0.85" />
      <circle cx="32" cy="46" r="4" fill={accent} />
      {/* Earpieces */}
      <path
        d="M20 20 L28 14 L36 14 L44 20"
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="20" cy="20" r="3.5" fill={accent} />
      <circle cx="44" cy="20" r="3.5" fill={accent} />
      {/* Right side tube */}
      <path
        d="M44 20 C44 20 50 22 50 30 C50 38 44 42 36 44"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
        fill="none"
        opacity="0.9"
      />
      {/* Plus/cross symbol */}
      <rect x="30" y="22" width="4" height="10" rx="2" fill={accent} opacity="0.8" />
      <rect x="25" y="25" width="14" height="4" rx="2" fill={accent} opacity="0.8" />
    </svg>
  );
}

const SVG_MAP: Record<PersonaId, (props: { color: string; accent: string }) => React.ReactNode> = {
  pathologist: PathologistSVG,
  qwen_vl: QwenVLSVG,
  researcher: ResearcherSVG,
  oncologist: OncologistSVG,
};

export default function AgentPersona({
  id,
  state = "idle",
  size = 80,
  showLabel = false,
}: AgentPersonaProps) {
  const cfg = PERSONAS[id];
  const SvgComp = SVG_MAP[id];

  const animationName =
    state === "speaking"
      ? "persona-speak"
      : state === "active"
      ? "persona-active"
      : "persona-idle";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
      <div
        style={{
          position: "relative",
          width: size,
          height: size,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {/* Pulse ring for active/speaking states */}
        {state !== "idle" && (
          <div
            style={{
              position: "absolute",
              inset: -4,
              borderRadius: "50%",
              border: `2px solid ${cfg.color}`,
              opacity: 0.6,
              animation: state === "speaking" ? "persona-pulse 0.7s ease-in-out infinite" : "persona-pulse 1.8s ease-in-out infinite",
            }}
          />
        )}
        {/* Main avatar circle */}
        <div
          style={{
            width: size,
            height: size,
            borderRadius: "50%",
            background: cfg.bgColor,
            border: `2px solid ${state !== "idle" ? cfg.color : cfg.color + "60"}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: size * 0.15,
            animation: `${animationName} ${state === "speaking" ? "0.8s" : "3s"} ease-in-out infinite`,
            boxShadow: state !== "idle" ? `0 0 20px ${cfg.color}40` : "none",
            transition: "box-shadow 0.3s ease, border-color 0.3s ease",
          }}
        >
          <SvgComp color={cfg.color} accent={cfg.accentColor} />
        </div>
      </div>

      {showLabel && (
        <div
          style={{
            fontSize: "0.72rem",
            fontWeight: 600,
            color: state !== "idle" ? cfg.color : "var(--text-muted)",
            textAlign: "center",
            maxWidth: size * 1.4,
            lineHeight: 1.3,
            transition: "color 0.3s ease",
          }}
        >
          {cfg.label}
        </div>
      )}

      <style>{`
        @keyframes persona-idle {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.025); }
        }
        @keyframes persona-active {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.04); }
        }
        @keyframes persona-speak {
          0%, 100% { transform: scale(1) rotate(-1deg); }
          25% { transform: scale(1.05) rotate(1deg); }
          75% { transform: scale(0.97) rotate(-0.5deg); }
        }
        @keyframes persona-pulse {
          0%, 100% { transform: scale(1); opacity: 0.6; }
          50% { transform: scale(1.15); opacity: 0.15; }
        }
      `}</style>
    </div>
  );
}
