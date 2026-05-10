"use client";

import { useEffect, useState, useCallback } from "react";
import { getVram, type VramInfo } from "@/lib/api";

// Colour per model component id
const SEGMENT_COLORS: Record<string, string> = {
  llama:    "#0891b2",
  qwen_vl:  "#22c55e",
  lora:     "#38bdf8",
  kv_cache: "#7c3aed",
  gigapath: "#0d9488",
  overhead: "#475569",
};

function segColor(id: string): string {
  return SEGMENT_COLORS[id] ?? "#14b8a6";
}

export default function VramBar({ compact = false }: { compact?: boolean }) {
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [error, setError] = useState(false);
  const [ready, setReady] = useState(false);

  const poll = useCallback(async () => {
    try {
      const v = await getVram();
      setVram(v);
      setError(false);
      setReady(true);
    } catch {
      setError(true);
    }
  }, []);

  useEffect(() => {
    poll();
    const iv = setInterval(poll, 2000);
    return () => clearInterval(iv);
  }, [poll]);

  const totalGb  = vram?.total_gb  ?? 192;
  const usedGb   = vram?.used_gb   ?? 0;
  const freeGb   = totalGb - usedGb;
  const pct      = totalGb > 0 ? Math.min((usedGb / totalGb) * 100, 100) : 0;
  const h100Pct  = (80 / totalGb) * 100;           // where H100 would cap out
  const overflow = Math.max(0, usedGb - 80);

  const components = (vram?.model_components ?? []).filter((c) => c.gb > 0);
  const compTotal  = components.reduce((s, c) => s + c.gb, 0);

  const pad       = compact ? "1rem"    : "1.5rem";
  const gap       = compact ? "0.9rem"  : "1.1rem";
  const barH      = compact ? "16px"    : "22px";
  const labelSz   = compact ? "0.8rem"  : "0.85rem";
  const tagSz     = compact ? "0.68rem" : "0.72rem";
  const titleSz   = compact ? "0.88rem" : "0.95rem";

  return (
    <div className="glass-card" style={{ padding: pad, display: "flex", flexDirection: "column", gap }}>

      {/* ── Header ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700, fontSize: titleSz, color: "var(--text-primary)" }}>
          VRAM · AMD MI300X
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <div className="pulse-dot pulse-dot-teal" />
          <span style={{ fontSize: "0.7rem", color: "var(--teal-light)" }}>
            {error ? "offline" : "live"}
          </span>
        </div>
      </div>

      {/* ── MI300X bar ── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
          <span style={{ fontSize: labelSz, fontWeight: 600, color: "var(--teal-light)" }}>
            MI300X — 192 GB HBM3
          </span>
          <span style={{ fontSize: labelSz, color: "var(--text-primary)", fontWeight: 700 }}>
            {usedGb.toFixed(1)} / {totalGb.toFixed(0)} GB
            <span style={{ color: "var(--text-muted)", fontWeight: 400, marginLeft: "0.4rem" }}>
              ({freeGb.toFixed(0)} GB free)
            </span>
          </span>
        </div>

        {/* Segmented bar */}
        <div
          style={{
            height: barH,
            background: "rgba(255,255,255,0.05)",
            borderRadius: 10,
            overflow: "hidden",
            border: "1px solid var(--teal-border)",
            display: "flex",
          }}
        >
          {ready && compTotal > 0
            ? components.map((c) => (
                <div
                  key={c.id}
                  title={`${c.label}: ${c.gb.toFixed(1)} GB`}
                  style={{
                    height: "100%",
                    width: `${(c.gb / totalGb) * 100}%`,
                    background: segColor(c.id),
                    transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                    flexShrink: 0,
                  }}
                />
              ))
            : (
              <div
                style={{
                  height: "100%",
                  width: ready ? `${pct}%` : "0%",
                  background: "linear-gradient(90deg, #0d9488, #14b8a6)",
                  borderRadius: 10,
                  transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                }}
              />
            )}
        </div>

        {/* Model legend */}
        {components.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem 0.75rem", marginTop: "0.55rem" }}>
            {components.map((c) => (
              <span key={c.id} style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: tagSz, color: "var(--text-muted)" }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: segColor(c.id), flexShrink: 0, display: "inline-block" }} />
                {c.label}
                <span style={{ fontFamily: "monospace", color: "var(--text-dim)" }}>{c.gb.toFixed(1)} GB</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* ── H100 comparison bar ── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.35rem" }}>
          <span style={{ fontSize: labelSz, fontWeight: 600, color: "var(--danger)" }}>
            H100 SXM5 — 80 GB limit
          </span>
          <span style={{
            fontSize: "0.7rem", fontWeight: 700,
            background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.4)",
            color: "var(--danger)", padding: "0.08rem 0.55rem", borderRadius: 20,
          }}>
            ✗ OOM {overflow > 0 ? `+${overflow.toFixed(0)} GB` : ""}
          </span>
        </div>

        <div
          style={{
            height: barH,
            background: "rgba(239,68,68,0.06)",
            borderRadius: 10,
            overflow: "hidden",
            border: "1px solid rgba(239,68,68,0.3)",
            display: "flex",
          }}
        >
          {/* Solid red up to 80 GB */}
          <div style={{
            width: `${h100Pct}%`,
            height: "100%",
            background: "linear-gradient(90deg, #dc2626, #ef4444)",
            flexShrink: 0,
          }} />
          {/* Overflow hatching */}
          <div style={{
            flex: 1,
            height: "100%",
            background: "repeating-linear-gradient(45deg,rgba(239,68,68,0.3),rgba(239,68,68,0.3) 4px,transparent 4px,transparent 10px)",
            borderLeft: "2px solid #ef4444",
            display: "flex",
            alignItems: "center",
            paddingLeft: "0.5rem",
          }}>
            <span style={{ fontSize: tagSz, color: "#fca5a5", fontWeight: 700, whiteSpace: "nowrap" }}>
              needs multi-GPU sharding
            </span>
          </div>
        </div>
      </div>

    </div>
  );
}
