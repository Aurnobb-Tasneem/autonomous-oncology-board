"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { getVram, type VramInfo, type VramModelComponent } from "@/lib/api";

const COMPONENT_COLORS: Record<string, string> = {
  gigapath: "#059669",
  llama_weights: "#0891b2",
  kv_cache: "#7c3aed",
  runtime: "#64748b",
};

function componentColor(id: string): string {
  return COMPONENT_COLORS[id] ?? "#14b8a6";
}

function sumComponents(components: VramModelComponent[] | null | undefined): number {
  if (!components?.length) return 0;
  return components.reduce((s, c) => s + Math.max(0, c.gb), 0);
}

function sumProcessRows(rows: { gb: number }[] | null | undefined): number {
  if (!rows?.length) return 0;
  return rows.reduce((s, r) => s + Math.max(0, r.gb), 0);
}

/** Backend reports per-process VRAM as decimal GB (bytes / 1e9). Convert to GiB for display next to rocm-smi totals. */
function decimalGbToGib(gb: number): number {
  return (gb * 1e9) / 1024 ** 3;
}

const PROCESS_BAR_COLORS: Record<string, string> = {
  ollama: "#0891b2",
  uvicorn: "#059669",
  python: "#059669",
  python3: "#059669",
  vllm: "#7c3aed",
  _other: "#64748b",
};

function processBarColor(processKey: string): string {
  return PROCESS_BAR_COLORS[processKey] ?? "#14b8a6";
}

export default function VramBar({ compact = false }: { compact?: boolean }) {
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [animIn, setAnimIn] = useState(false);

  const fetchVram = useCallback(async () => {
    try {
      const v = await getVram();
      setVram(v);
      setFetchError(null);
      setAnimIn(true);
    } catch (e) {
      setVram(null);
      setFetchError(e instanceof Error ? e.message : "VRAM request failed");
    }
  }, []);

  useEffect(() => {
    fetchVram();
    const iv = setInterval(fetchVram, 2000);
    return () => clearInterval(iv);
  }, [fetchVram]);

  const useGib =
    vram != null &&
    typeof vram.total_gib === "number" &&
    vram.total_gib > 0 &&
    typeof vram.used_gib === "number";
  const totalCap = useGib ? vram!.total_gib! : vram && vram.total_gb > 0 ? vram.total_gb : 192;
  const used = useGib ? vram!.used_gib! : vram?.used_gb ?? 0;
  const pctDisplay =
    useGib && typeof vram!.percent_gib === "number"
      ? vram!.percent_gib!
      : vram
        ? Math.min((used / totalCap) * 100, 100)
        : 0;
  const mi300xPct = vram ? Math.min(pctDisplay, 100) : 0;

  const useMeasuredBar =
    vram?.source === "rocm-smi" && (vram.processes_display?.length ?? 0) > 0;
  const procSum = useMemo(
    () => sumProcessRows(vram?.processes_display ?? null),
    [vram?.processes_display],
  );
  const compSum = useMemo(() => sumComponents(vram?.model_components ?? null), [vram?.model_components]);
  const barSegmentSum = useMeasuredBar ? procSum : compSum;

  const pad = compact ? "1rem" : "1.5rem";
  const headerMb = compact ? "0.85rem" : "1.25rem";
  const barH = compact ? "14px" : "20px";
  const titleSize = compact ? "0.88rem" : "0.95rem";
  const smallSize = compact ? "0.7rem" : "0.75rem";
  const labelSize = compact ? "0.78rem" : "0.82rem";
  const breakdownSize = compact ? "0.68rem" : "0.72rem";

  const components = vram?.model_components?.filter((c) => c.gb > 0) ?? [];

  return (
    <div className="glass-card" style={{ padding: pad }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: headerMb }}>
        <h3 style={{ fontWeight: 700, fontSize: titleSize, color: "var(--text-primary)" }}>
          Live VRAM Usage
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <div className="pulse-dot pulse-dot-teal" />
          <span style={{ fontSize: smallSize, color: "var(--teal-light)" }}>Live</span>
          {vram?.source === "mock" && (
            <span
              style={{
                fontSize: compact ? "0.66rem" : "0.7rem",
                color: "var(--text-muted)",
                marginLeft: "0.25rem",
              }}
            >
              (backend mock — rocm-smi unavailable)
            </span>
          )}
          {vram?.source === "rocm-smi" && (
            <span
              style={{
                fontSize: compact ? "0.66rem" : "0.7rem",
                color: "var(--teal-light)",
                marginLeft: "0.25rem",
              }}
            >
              rocm-smi
            </span>
          )}
        </div>
      </div>

      {fetchError && (
        <div
          style={{
            marginBottom: "0.75rem",
            padding: "0.5rem 0.65rem",
            borderRadius: 8,
            fontSize: breakdownSize,
            color: "#fecaca",
            background: "rgba(127,29,29,0.35)",
            border: "1px solid rgba(248,113,113,0.4)",
          }}
        >
          VRAM API unreachable: {fetchError}. Check Next.js <code style={{ fontSize: "0.85em" }}>BACKEND_INTERNAL_URL</code> /{" "}
          <code style={{ fontSize: "0.85em" }}>NEXT_PUBLIC_API_URL</code>.
        </div>
      )}

      <div style={{ marginBottom: compact ? "1rem" : "1.25rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: labelSize, fontWeight: 600, color: "var(--teal-light)" }}>
            AMD MI300X · {totalCap.toFixed(1)} {useGib ? "GiB" : "GB"} HBM3
          </span>
          <span style={{ fontSize: labelSize, color: "var(--text-muted)" }}>
            {vram
              ? `${used.toFixed(2)} / ${totalCap.toFixed(2)} ${useGib ? "GiB" : "GB"}`
              : "—"}
          </span>
        </div>

        <div
          style={{
            height: barH,
            background: "rgba(255,255,255,0.05)",
            borderRadius: "10px",
            overflow: "hidden",
            border: "1px solid var(--teal-border)",
            position: "relative",
          }}
        >
          <div
            style={{
              height: "100%",
              width: animIn ? `${mi300xPct}%` : "0%",
              background: "linear-gradient(90deg, #0d9488, #14b8a6)",
              borderRadius: "10px",
              transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
              boxShadow: "0 0 12px rgba(13,148,136,0.5)",
              display: "flex",
              overflow: "hidden",
            }}
          >
            {barSegmentSum > 0 &&
              useMeasuredBar &&
              vram!.processes_display!.map((row) => (
                <div
                  key={row.process}
                  title={`${row.label}: ${useGib ? decimalGbToGib(row.gb).toFixed(2) : row.gb.toFixed(1)} ${useGib ? "GiB" : "GB"} (measured)`}
                  style={{
                    height: "100%",
                    width: `${(Math.max(0, row.gb) / barSegmentSum) * 100}%`,
                    background: processBarColor(row.process),
                    opacity: 0.92,
                    transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                  }}
                />
              ))}
            {barSegmentSum > 0 &&
              !useMeasuredBar &&
              components.map((c) => (
                <div
                  key={c.id}
                  title={`${c.label}: ${c.gb.toFixed(1)} GB (est.)`}
                  style={{
                    height: "100%",
                    width: `${(Math.max(0, c.gb) / barSegmentSum) * 100}%`,
                    background: componentColor(c.id),
                    opacity: 0.92,
                    transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                  }}
                />
              ))}
          </div>
        </div>

        {barSegmentSum > 0 && useMeasuredBar && (
          <div
            style={{
              fontSize: compact ? "0.62rem" : "0.65rem",
              color: "var(--text-muted)",
              marginTop: "0.28rem",
            }}
          >
            Bar segments: measured VRAM per GPU process (<span style={{ fontFamily: "monospace" }}>rocm-smi --showpids</span>)
          </div>
        )}
        {barSegmentSum > 0 && !useMeasuredBar && compSum > 0 && (
          <div
            style={{
              fontSize: compact ? "0.62rem" : "0.65rem",
              color: "var(--text-muted)",
              marginTop: "0.28rem",
            }}
          >
            Strip inside bar: estimated split (no per-process sample — GigaPath · Llama · KV · overhead)
          </div>
        )}

        {vram?.ollama_model && (
          <div style={{ fontSize: breakdownSize, color: "var(--text-muted)", marginTop: "0.35rem" }}>
            Ollama model: <span style={{ fontFamily: "monospace", color: "var(--text-dim)" }}>{vram.ollama_model}</span>
          </div>
        )}

        {vram?.processes_display && vram.processes_display.length > 0 && (
          <div style={{ marginTop: "0.65rem" }}>
            <div
              style={{
                fontSize: compact ? "0.66rem" : "0.7rem",
                fontWeight: 700,
                color: "var(--text-primary)",
                marginBottom: "0.35rem",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              Measured (GPU processes)
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
              {vram.processes_display.map((row) => (
                <div
                  key={row.process}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "baseline",
                    fontSize: breakdownSize,
                    gap: "0.5rem",
                  }}
                >
                  <span style={{ color: "var(--text-muted)" }}>{row.label}</span>
                  <span style={{ fontFamily: "monospace", color: "var(--teal-light)", fontWeight: 600 }}>
                    {useGib ? `${decimalGbToGib(row.gb).toFixed(2)} GiB` : `${row.gb.toFixed(1)} GB`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {vram?.model_components && vram.model_components.some((c) => c.gb > 0) && (
          <div style={{ marginTop: "0.65rem" }}>
            <div
              style={{
                fontSize: compact ? "0.66rem" : "0.7rem",
                fontWeight: 700,
                color: "var(--text-primary)",
                marginBottom: "0.35rem",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              Estimated model footprint
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.55rem" }}>
              {vram.model_components
                .filter((c) => c.gb > 0)
                .map((c) => (
                  <span key={c.id} style={{ fontSize: breakdownSize, color: "var(--text-muted)" }}>
                    <span style={{ color: componentColor(c.id) }}>●</span> {c.label}{" "}
                    <span style={{ color: "var(--text-dim)", fontFamily: "monospace" }}>{c.gb.toFixed(1)} GB</span>
                  </span>
                ))}
            </div>
          </div>
        )}

        {!vram?.processes_display?.length && vram?.model_breakdown && (
          <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
            {[
              { label: "GigaPath", val: vram.model_breakdown.gigapath_gb },
              { label: "Llama", val: vram.model_breakdown.llama_gb },
              { label: "KV cache", val: vram.model_breakdown.kv_cache_gb },
              { label: "Runtime", val: vram.model_breakdown.runtime_overhead_gb ?? undefined },
            ]
              .filter((m) => m.val != null && (m.val as number) > 0)
              .map((m) => (
                <span key={m.label} style={{ fontSize: breakdownSize, color: "var(--text-muted)" }}>
                  <span style={{ color: "var(--teal-light)" }}>●</span> {m.label}{" "}
                  <span style={{ color: "var(--text-dim)" }}>{m.val} GB</span>
                </span>
              ))}
          </div>
        )}
      </div>

      {/* H100 OOM comparison */}
      <div style={{ marginTop: compact ? "1rem" : "1.25rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: labelSize, fontWeight: 600, color: "var(--danger)" }}>
            NVIDIA H100 SXM5 · 80 GB limit
          </span>
          <span
            style={{
              fontSize: smallSize,
              background: "rgba(239,68,68,0.15)",
              border: "1px solid rgba(239,68,68,0.4)",
              color: "var(--danger)",
              padding: "0.1rem 0.5rem",
              borderRadius: "20px",
              fontWeight: 700,
            }}
          >
            ✗ OOM
          </span>
        </div>

        {/* H100 bar: filled to 80/192 of width, then red OOM stripe */}
        <div
          style={{
            height: barH,
            background: "rgba(239,68,68,0.06)",
            borderRadius: "10px",
            overflow: "hidden",
            border: "1px solid rgba(239,68,68,0.3)",
            position: "relative",
            display: "flex",
          }}
        >
          {/* Filled portion that "fits" in H100 (≈42% = 80/192) */}
          <div style={{ width: "41.67%", height: "100%", background: "linear-gradient(90deg,#dc2626,#ef4444)", flexShrink: 0 }} />
          {/* Overflow hatching */}
          <div
            style={{
              flex: 1,
              height: "100%",
              background: "repeating-linear-gradient(45deg,rgba(239,68,68,0.25),rgba(239,68,68,0.25) 4px,transparent 4px,transparent 10px)",
              borderLeft: "2px solid #ef4444",
              display: "flex",
              alignItems: "center",
              paddingLeft: "0.5rem",
            }}
          >
            <span style={{ fontSize: compact ? "0.6rem" : "0.66rem", color: "#fca5a5", fontWeight: 700, whiteSpace: "nowrap" }}>
              +{used > 0 ? `${(used - 80).toFixed(1)} GiB overflow` : "≥5 GiB overflow"} — needs sharding
            </span>
          </div>
        </div>

        {/* Math breakdown */}
        <div
          style={{
            marginTop: "0.5rem",
            padding: "0.5rem 0.75rem",
            background: "rgba(239,68,68,0.05)",
            border: "1px solid rgba(239,68,68,0.2)",
            borderRadius: "8px",
            fontSize: breakdownSize,
            color: "rgba(252,165,165,0.85)",
            lineHeight: 1.7,
          }}
        >
          <strong style={{ color: "#fca5a5" }}>Why H100 fails:</strong>{" "}
          Llama 3.3 70B (Q4_K_S) <span style={{ fontFamily: "monospace" }}>~40 GB</span>
          {" "}+ GigaPath ViT-Giant <span style={{ fontFamily: "monospace" }}>~3 GB</span>
          {" "}+ LoRA Specialists <span style={{ fontFamily: "monospace" }}>~22 GB</span>
          {" "}+ KV Cache <span style={{ fontFamily: "monospace" }}>~20 GB</span>
          {" "}= <span style={{ fontFamily: "monospace", color: "#f87171", fontWeight: 700 }}>~85 GB</span>
          {" "}— 5 GB over the H100 hard limit, before a single token generates.{" "}
          The MI300X has <span style={{ fontFamily: "monospace", color: "#4ade80", fontWeight: 700 }}>192 GB</span> — the only GPU where this architecture runs on a single card.
        </div>
      </div>

      <div
        style={{
          marginTop: "0.85rem",
          paddingTop: "0.65rem",
          borderTop: "1px solid var(--border)",
          fontSize: breakdownSize,
          color: "var(--text-muted)",
          textAlign: "center",
        }}
      >
        Live VRAM from <span style={{ fontFamily: "monospace" }}>rocm-smi --showpids</span> · polling every 2s
        {useGib ? " · values in GiB (1 GiB = 1024³ bytes)" : ""}
      </div>
    </div>
  );
}
