"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth } from "@/lib/api";

export default function NavBar() {
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking");

  useEffect(() => {
    const check = async () => {
      try {
        const h = await getHealth();
        setApiStatus(h.status === "ok" ? "online" : "offline");
      } catch {
        setApiStatus("offline");
      }
    };
    check();
    const iv = setInterval(check, 15000);
    return () => clearInterval(iv);
  }, []);

  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        background: "rgba(10,22,40,0.92)",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid var(--border-teal)",
        padding: "0 2rem",
        height: "60px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      {/* Logo */}
      <Link
        href="/"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          textDecoration: "none",
          color: "var(--text-primary)",
        }}
      >
        <span style={{ fontSize: "1.4rem" }}>🔬</span>
        <div>
          <span style={{ fontWeight: 700, fontSize: "1rem", color: "var(--teal-light)" }}>
            AOB
          </span>
          <span style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginLeft: "0.4rem" }}>
            Autonomous Oncology Board
          </span>
        </div>
      </Link>

      {/* Nav links */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.1rem" }}>
        {[
          { href: "/specialists", label: "Specialists" },
          { href: "/benchmark", label: "Benchmark" },
          { href: "/concurrent", label: "Concurrent" },
          { href: "/story", label: "Story" },
        ].map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            style={{
              padding: "0.3rem 0.7rem",
              borderRadius: "6px",
              fontSize: "0.82rem",
              fontWeight: 500,
              color: "var(--text-muted)",
              textDecoration: "none",
              transition: "color 0.15s ease",
            }}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.color = "var(--text-primary)"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.color = "var(--text-muted)"; }}
          >
            {label}
          </Link>
        ))}
      </div>

      {/* Right side badges */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        {/* AMD badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            background: "rgba(13,148,136,0.12)",
            border: "1px solid var(--teal-border)",
            borderRadius: "20px",
            padding: "0.25rem 0.75rem",
            fontSize: "0.8rem",
            fontWeight: 600,
            color: "var(--teal-light)",
          }}
        >
          <span>▲</span>
          <span>AMD MI300X · 192 GB</span>
        </div>

        {/* API status */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            fontSize: "0.8rem",
            color: apiStatus === "online" ? "var(--success)" : apiStatus === "offline" ? "var(--danger)" : "var(--text-muted)",
          }}
        >
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: apiStatus === "online" ? "var(--success)" : apiStatus === "offline" ? "var(--danger)" : "var(--text-muted)",
              boxShadow: apiStatus === "online" ? "0 0 6px var(--success)" : "none",
              animation: apiStatus === "online" ? "pulse-dot 1.8s ease-in-out infinite" : "none",
            }}
          />
          <span>API {apiStatus === "checking" ? "..." : apiStatus}</span>
        </div>
      </div>
    </nav>
  );
}
