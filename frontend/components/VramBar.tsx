"use client";

import { useEffect, useState, useCallback } from "react";
import { getVram, type VramInfo } from "@/lib/api";

const H100_MAX = 80;
const MI300X_MAX = 192;

export default function VramBar() {
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [animIn, setAnimIn] = useState(false);

  const fetchVram = useCallback(async () => {
    try {
      const v = await getVram();
      setVram(v);
      setAnimIn(true);
    } catch {
      // API not reachable — show mock
      setVram({
        used_gb: 52.4,
        total_gb: 192,
        free_gb: 139.6,
        percent_used: 27.3,
        model_breakdown: { gigapath_gb: 2.1, llama_gb: 38.0, kv_cache_gb: 12.3 },
        source: "mock",
      });
      setAnimIn(true);
    }
  }, []);

  useEffect(() => {
    fetchVram();
    const iv = setInterval(fetchVram, 2000);
    return () => clearInterval(iv);
  }, [fetchVram]);

  const mi300xPct = vram ? Math.min((vram.used_gb / MI300X_MAX) * 100, 100) : 0;
  const h100Pct = 100; // always OOM

  return (
    <div className="glass-card" style={{ padding: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <h3 style={{ fontWeight: 700, fontSize: "0.95rem", color: "var(--text-primary)" }}>
          Live VRAM Usage
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <div className="pulse-dot pulse-dot-teal" />
          <span style={{ fontSize: "0.75rem", color: "var(--teal-light)" }}>Live</span>
          {vram?.source === "mock" && (
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginLeft: "0.25rem" }}>(demo)</span>
          )}
        </div>
      </div>

      {/* MI300X bar */}
      <div style={{ marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--teal-light)" }}>
            AMD MI300X 192 GB HBM3
          </span>
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
            {vram ? `${vram.used_gb.toFixed(1)} / ${MI300X_MAX} GB` : "—"}
          </span>
        </div>
        {/* Track */}
        <div
          style={{
            height: "20px",
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
            }}
          />
        </div>
        {/* Model breakdown */}
        {vram?.model_breakdown && (
          <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.35rem", flexWrap: "wrap" }}>
            {[
              { label: "GigaPath", val: vram.model_breakdown.gigapath_gb },
              { label: "Llama 70B", val: vram.model_breakdown.llama_gb },
              { label: "KV Cache", val: vram.model_breakdown.kv_cache_gb },
            ].map((m) => (
              <span key={m.label} style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                <span style={{ color: "var(--teal-light)" }}>●</span> {m.label}{" "}
                <span style={{ color: "var(--text-dim)" }}>{m.val} GB</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* H100 bar — always OOM */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--danger)" }}>
            H100 SXM5 80 GB
          </span>
          <span
            style={{
              fontSize: "0.75rem",
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
            height: "20px",
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
            <span style={{ fontSize: "0.7rem", color: "white", fontWeight: 700, letterSpacing: "0.05em" }}>
              OUT OF MEMORY — Cannot load both models
            </span>
          </div>
        </div>
        <p style={{ fontSize: "0.72rem", color: "rgba(239,68,68,0.7)", marginTop: "0.3rem" }}>
          GigaPath (2.1 GB) + Llama 3.3 70B (38 GB) = 52+ GB needed simultaneously
        </p>
      </div>

      {/* Footer note */}
      <div
        style={{
          marginTop: "1rem",
          paddingTop: "0.75rem",
          borderTop: "1px solid var(--border)",
          fontSize: "0.72rem",
          color: "var(--text-muted)",
          textAlign: "center",
        }}
      >
        MI300X's 192 GB HBM3 unified memory enables permanent model loading — no swapping.
      </div>
    </div>
  );
}
