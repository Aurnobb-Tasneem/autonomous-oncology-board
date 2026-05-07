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

export default function VramBar({ compact = false }: { compact?: boolean }) {
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [animIn, setAnimIn] = useState(false);

  const fetchVram = useCallback(async () => {
    try {
      const v = await getVram();
      setVram(v);
      setAnimIn(true);
    } catch {
      setVram({
        used_gb: 102.3,
        total_gb: 206,
        free_gb: 103.7,
        used_gib: 95.25,
        total_gib: 192,
        free_gib: 96.75,
        percent_used: 49.7,
        percent_gib: 49.7,
        source: "mock",
        model_breakdown: {
          gigapath_gb: 3.2,
          llama_gb: 140,
          kv_cache_gb: 28,
          runtime_overhead_gb: 4,
        },
        model_components: [
          { id: "gigapath", label: "GigaPath (ViT, est.)", gb: 3.2 },
          { id: "llama_weights", label: "Llama 70B weights (FP16 est.)", gb: 140 },
          { id: "kv_cache", label: "KV cache (inside Ollama, est.)", gb: 28 },
          { id: "runtime", label: "API / torch overhead (est.)", gb: 4 },
        ],
        processes_display: [
          { process: "ollama", label: "Ollama — Llama 70B + KV cache", gb: 168 },
          { process: "uvicorn", label: "FastAPI / GigaPath (uvicorn)", gb: 22 },
        ],
      });
      setAnimIn(true);
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

  const compSum = useMemo(() => sumComponents(vram?.model_components ?? null), [vram?.model_components]);

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
              (demo / offline)
            </span>
          )}
        </div>
      </div>

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
            {compSum > 0 &&
              components.map((c) => (
                <div
                  key={c.id}
                  title={`${c.label}: ${c.gb.toFixed(1)} GB (est.)`}
                  style={{
                    height: "100%",
                    width: `${(Math.max(0, c.gb) / compSum) * 100}%`,
                    background: componentColor(c.id),
                    opacity: 0.92,
                    transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                  }}
                />
              ))}
          </div>
        </div>

        {compSum > 0 && (
          <div
            style={{
              fontSize: compact ? "0.62rem" : "0.65rem",
              color: "var(--text-muted)",
              marginTop: "0.28rem",
            }}
          >
            Strip inside bar: estimated split (GigaPath · Llama weights · KV · overhead)
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
                    {row.gb.toFixed(1)} GB
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

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: labelSize, fontWeight: 600, color: "var(--danger)" }}>
            H100 SXM5 80 GB
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
        <div
          style={{
            height: barH,
            background: "rgba(239,68,68,0.08)",
            borderRadius: "10px",
            overflow: "hidden",
            border: "1px solid rgba(239,68,68,0.3)",
          }}
        >
          <div
            style={{
              height: "100%",
              width: "100%",
              background: "linear-gradient(90deg, #dc2626, #ef4444)",
              borderRadius: "10px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontSize: compact ? "0.64rem" : "0.7rem",
                color: "white",
                fontWeight: 700,
                letterSpacing: "0.05em",
              }}
            >
              OUT OF MEMORY — Cannot load full stack
            </span>
          </div>
        </div>
        <p style={{ fontSize: breakdownSize, color: "rgba(239,68,68,0.7)", marginTop: "0.3rem" }}>
          FP16 70B weights (~140 GB) + GigaPath + KV cache exceeds a single 80 GB H100 without
          multi-GPU sharding.
        </p>
      </div>

      <div
        style={{
          marginTop: "1rem",
          paddingTop: "0.75rem",
          borderTop: "1px solid var(--border)",
          fontSize: breakdownSize,
          color: "var(--text-muted)",
          textAlign: "center",
        }}
      >
        Process rows come from <span style={{ fontFamily: "monospace" }}>rocm-smi --showpids</span>. Model
        lines use declared Llama weight budget + measured Ollama RSS for KV estimate.
      </div>
    </div>
  );
}
