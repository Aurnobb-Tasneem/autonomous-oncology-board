"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import NavBar from "@/components/NavBar";
import VramBar from "@/components/VramBar";
import CaseCard from "@/components/CaseCard";
import { analyzeImages, getDemoCases, type DemoCase } from "@/lib/api";

const STATS = [
  { value: "192 GB", label: "HBM3 Unified Memory" },
  { value: "3", label: "AI Agents" },
  { value: "≤3", label: "Debate Rounds" },
  { value: "~60s", label: "Full Pipeline" },
];

export default function HomePage() {
  const [demos, setDemos] = useState<DemoCase[]>([]);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const router = useRouter();

  useEffect(() => {
    getDemoCases().then(setDemos).catch(() => {
      // Fallback static demo list
      setDemos([
        { case_name: "lung_adenocarcinoma", tissue_type: "lung_adenocarcinoma", description: "67M, non-smoker, CT shows 3.2cm nodule", metadata: {} },
        { case_name: "colon_adenocarcinoma", tissue_type: "colon_adenocarcinoma", description: "58F, T3N1M0, post-colonoscopy biopsy", metadata: {} },
        { case_name: "lung_squamous_cell", tissue_type: "lung_squamous_cell_carcinoma", description: "72M, 50 pack-year smoker, central mass", metadata: {} },
      ]);
    });
  }, []);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) {
      setUploadFiles([]);
      return;
    }
    const next = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (next.length === 0) {
      setUploadError("Please add PNG or JPG patch images.");
      setUploadFiles([]);
      return;
    }
    setUploadError(null);
    setUploadFiles(next);
  };

  const handleUpload = async () => {
    if (uploadFiles.length === 0) {
      setUploadError("Add at least one patch image to continue.");
      return;
    }
    setUploadLoading(true);
    setUploadError(null);
    try {
      const resp = await analyzeImages(uploadFiles);
      router.push(`/analyze/${resp.job_id}`);
    } catch (e) {
      setUploadError("Upload failed — is the API running?");
      setUploadLoading(false);
    }
  };

  return (
    <>
      <NavBar />
      <main style={{ minHeight: "calc(100vh - 60px)" }}>

        {/* ── Hero Section ─────────────────────────────────────────────── */}
        <section
          style={{
            maxWidth: "1200px",
            margin: "0 auto",
            padding: "4rem 2rem 3rem",
            display: "grid",
            gridTemplateColumns: "1fr 380px",
            gap: "3rem",
            alignItems: "start",
          }}
        >
          {/* Left: headline + CTA */}
          <div>
            {/* Badge */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                background: "rgba(13,148,136,0.1)",
                border: "1px solid var(--teal-border)",
                borderRadius: "20px",
                padding: "0.3rem 0.85rem",
                fontSize: "0.8rem",
                color: "var(--teal-light)",
                fontWeight: 600,
                marginBottom: "1.5rem",
              }}
            >
              <span>🏆</span>
              <span>AMD MI300X Hackathon · lablab.ai</span>
            </div>

            {/* Headline */}
            <h1
              style={{
                fontSize: "clamp(2rem, 4vw, 3.2rem)",
                fontWeight: 800,
                lineHeight: 1.15,
                color: "var(--text-primary)",
                marginBottom: "1.25rem",
              }}
            >
              3-Agent AI Tumour Board
              <br />
              <span className="text-teal-glow">Powered by AMD MI300X</span>
            </h1>

            {/* Subtitle */}
            <p
              style={{
                fontSize: "1.05rem",
                color: "var(--text-muted)",
                lineHeight: 1.7,
                maxWidth: "560px",
                marginBottom: "2.5rem",
              }}
            >
              The only GPU with <strong style={{ color: "var(--text-secondary)" }}>192 GB unified memory</strong> — running{" "}
              <strong style={{ color: "var(--text-secondary)" }}>GigaPath ViT-Giant + Llama 3.3 70B</strong> simultaneously,
              with multi-round agent debate grounded in NCCN 2024 guidelines.
              No model swapping. No OOM. Just results.
            </p>

            {/* Demo case cards */}
            <div>
              <p
                style={{
                  fontSize: "0.82rem",
                  fontWeight: 600,
                  color: "var(--text-muted)",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  marginBottom: "0.85rem",
                }}
              >
                ▶ Run a Demo Case — one click, no setup
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: "0.85rem" }}>
                {demos.map((d) => (
                  <CaseCard key={d.case_name} demo={d} />
                ))}
              </div>
            </div>
          </div>

          {/* Right: VRAM widget */}
          <div style={{ position: "sticky", top: "80px" }}>
            <VramBar />
          </div>
        </section>

        {/* ── Stats bar ──────────────────────────────────────────────────── */}
        <section
          style={{
            borderTop: "1px solid var(--border)",
            borderBottom: "1px solid var(--border)",
            background: "rgba(13,20,40,0.5)",
            padding: "1.5rem 2rem",
          }}
        >
          <div
            style={{
              maxWidth: "1200px",
              margin: "0 auto",
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "1rem",
              textAlign: "center",
            }}
          >
            {STATS.map((s) => (
              <div key={s.label}>
                <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--teal-light)", lineHeight: 1 }}>
                  {s.value}
                </div>
                <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── How it works ────────────────────────────────────────────────── */}
        <section style={{ maxWidth: "1200px", margin: "0 auto", padding: "3rem 2rem" }}>
          <h2
            style={{
              fontSize: "1.4rem",
              fontWeight: 700,
              color: "var(--text-primary)",
              marginBottom: "1.5rem",
              textAlign: "center",
            }}
          >
            How the Board Works
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1.25rem" }}>
            {[
              {
                icon: "🔬",
                label: "Pathologist",
                sub: "GigaPath ViT-Giant 1.1B",
                desc: "Analyses histopathology patches. Generates attention heatmaps. Runs MC Dropout uncertainty. Extracts 8 biomarker scores.",
                color: "#0d9488",
              },
              {
                icon: "📚",
                label: "Researcher",
                sub: "RAG + NCCN 2024",
                desc: "Retrieves evidence from NCCN guidelines. Challenges draft plan. Flags missing molecular tests and contraindicated regimens.",
                color: "#7c3aed",
              },
              {
                icon: "👨‍⚕️",
                label: "Oncologist",
                sub: "Llama 3.3 70B",
                desc: "Synthesises pathology + evidence into a full management plan. Revises under critique. Finalises with consensus score ≥70/100.",
                color: "#0891b2",
              },
            ].map((agent) => (
              <div
                key={agent.label}
                className="glass-card"
                style={{ padding: "1.5rem", borderColor: `${agent.color}40` }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>{agent.icon}</div>
                <div style={{ fontWeight: 700, fontSize: "1rem", color: agent.color, marginBottom: "0.2rem" }}>
                  {agent.label}
                </div>
                <div style={{ fontSize: "0.77rem", color: "var(--text-muted)", marginBottom: "0.75rem" }}>
                  {agent.sub}
                </div>
                <p style={{ fontSize: "0.83rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                  {agent.desc}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Debate pipeline ─────────────────────────────────────────────── */}
        <section
          style={{
            maxWidth: "1200px",
            margin: "0 auto",
            padding: "0 2rem 4rem",
            textAlign: "center",
          }}
        >
          <div
            className="glass-card"
            style={{
              padding: "2rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "0.75rem",
              flexWrap: "wrap",
              fontSize: "0.88rem",
              color: "var(--text-muted)",
            }}
          >
            {[
              "🔬 Pathologist",
              "→",
              "📚 Researcher",
              "→",
              "👨‍⚕️ Oncologist (draft)",
              "→",
              "⚖️ Debate Loop (×3)",
              "→",
              "✅ Consensus ≥70",
              "→",
              "📊 Final Report",
            ].map((step, i) => (
              <span
                key={i}
                style={{
                  color: step === "→" ? "var(--text-muted)" : step.startsWith("✅") ? "var(--success)" : "var(--text-secondary)",
                  fontWeight: step === "→" ? 400 : 600,
                }}
              >
                {step}
              </span>
            ))}
          </div>
        </section>

        {/* ── Upload your own ─────────────────────────────────────────────── */}
        <section style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 2rem 3rem" }}>
          <div
            className="glass-card"
            style={{ padding: "1.75rem", display: "flex", flexDirection: "column", gap: "1rem" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  Upload Your Own Patches
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
                  Optional: submit 224x224 histology patches (PNG/JPG). Demo cases work best for judges.
                </div>
              </div>
              <button
                className="btn-ghost"
                style={{ padding: "0.45rem 1rem", fontSize: "0.8rem" }}
                onClick={() => setUploadOpen((o) => !o)}
              >
                {uploadOpen ? "Hide uploader" : "Show uploader"}
              </button>
            </div>

            {uploadOpen && (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragActive(false);
                    handleFiles(e.dataTransfer.files);
                  }}
                  style={{
                    border: `1px dashed ${dragActive ? "var(--teal-light)" : "var(--border-teal)"}`,
                    borderRadius: "12px",
                    padding: "1.5rem",
                    textAlign: "center",
                    background: dragActive ? "rgba(13,148,136,0.08)" : "rgba(13,20,40,0.4)",
                    color: "var(--text-muted)",
                  }}
                >
                  <div style={{ fontSize: "0.95rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                    Drag & drop patch images here
                  </div>
                  <div style={{ fontSize: "0.78rem", marginBottom: "0.85rem" }}>
                    PNG or JPG · Multiple files supported
                  </div>
                  <button
                    className="btn-teal"
                    style={{ padding: "0.5rem 1.25rem", fontSize: "0.85rem" }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Select files
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    style={{ display: "none" }}
                    onChange={(e) => handleFiles(e.target.files)}
                  />
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    {uploadFiles.length > 0 ? `${uploadFiles.length} file${uploadFiles.length !== 1 ? "s" : ""} ready` : "No files selected"}
                  </div>
                  <div style={{ display: "flex", gap: "0.65rem" }}>
                    <button
                      className="btn-ghost"
                      style={{ padding: "0.45rem 1rem", fontSize: "0.8rem" }}
                      onClick={() => {
                        setUploadFiles([]);
                        setUploadError(null);
                      }}
                    >
                      Clear
                    </button>
                    <button
                      className="btn-teal"
                      style={{ padding: "0.5rem 1.25rem", fontSize: "0.85rem", opacity: uploadLoading ? 0.8 : 1 }}
                      disabled={uploadLoading}
                      onClick={handleUpload}
                    >
                      {uploadLoading ? "Starting..." : "Run Analysis"}
                    </button>
                  </div>
                </div>

                {uploadError && (
                  <div style={{ fontSize: "0.8rem", color: "var(--danger)" }}>
                    {uploadError}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* Footer */}
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "1.5rem 2rem",
            textAlign: "center",
            fontSize: "0.78rem",
            color: "var(--text-muted)",
          }}
        >
          ⚠️ AI research tool for the AMD MI300X Hackathon — NOT for clinical use.{" "}
          <a
            href="https://github.com/Aurnobb-Tasneem/autonomous-oncology-board"
            target="_blank"
            rel="noreferrer"
            style={{ color: "var(--teal-light)", textDecoration: "none" }}
          >
            GitHub
          </a>
        </footer>
      </main>
    </>
  );
}
