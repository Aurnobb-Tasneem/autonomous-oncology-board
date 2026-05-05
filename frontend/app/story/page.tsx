"use client";

import { useEffect, useRef, useState } from "react";
import NavBar from "@/components/NavBar";
import AgentPersona, { type PersonaId, type PersonaState } from "@/components/AgentPersona";
import RevisionDiff from "@/components/RevisionDiff";
import H100Simulator from "@/components/H100Simulator";
import Leaderboard from "@/components/Leaderboard";
import { FALLBACK_ABLATION_INLINE } from "./fallback-data";

// Sections definition
const SECTIONS = [
  { id: "problem", number: "01", title: "The Problem" },
  { id: "team", number: "02", title: "Three Agents Enter" },
  { id: "pathologist", number: "03", title: "The Pathologist Sees" },
  { id: "researcher", number: "04", title: "The Researcher Cites" },
  { id: "debate", number: "05", title: "The Debate" },
  { id: "hardware", number: "06", title: "The Hardware Proof" },
  { id: "benchmark", number: "07", title: "The Numbers" },
];

function useIntersection(ref: React.RefObject<HTMLElement | null>, threshold = 0.45) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref, threshold]);
  return visible;
}

function Section({
  id,
  number,
  title,
  children,
  active,
}: {
  id: string;
  number: string;
  title: string;
  children: React.ReactNode;
  active: boolean;
}) {
  return (
    <section
      id={id}
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        padding: "6rem 2rem",
        scrollSnapAlign: "start",
        position: "relative",
        borderBottom: "1px solid rgba(148,163,184,0.06)",
      }}
    >
      <div style={{ maxWidth: "1000px", margin: "0 auto", width: "100%" }}>
        {/* Section number */}
        <div
          style={{
            fontSize: "0.68rem",
            fontWeight: 700,
            color: active ? "var(--teal-light)" : "var(--text-muted)",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            marginBottom: "0.75rem",
            transition: "color 0.4s ease",
          }}
        >
          {number} / {title}
        </div>
        <div
          style={{
            opacity: active ? 1 : 0.35,
            transform: active ? "translateY(0)" : "translateY(20px)",
            transition: "opacity 0.6s ease, transform 0.6s ease",
          }}
        >
          {children}
        </div>
      </div>
    </section>
  );
}

export default function StoryPage() {
  const refs = SECTIONS.map(() => useRef<HTMLElement>(null));
  const visibles = refs.map((ref) => useIntersection(ref as React.RefObject<HTMLElement>, 0.3));
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    const last = visibles.lastIndexOf(true);
    if (last >= 0) setActiveIdx(last);
  }, [visibles]);

  // Debated plan text example
  const beforeRevision =
    "First-line: Platinum-based chemotherapy (cisplatin + pemetrexed).";
  const afterRevision =
    "First-line: Osimertinib 80 mg/day (EGFR-mutant, NCCN Category 1, FLAURA trial). Platinum-based chemotherapy if EGFR negative.";

  return (
    <>
      <NavBar />

      <div
        style={{
          scrollSnapType: "y mandatory",
          overflowY: "scroll",
          height: "calc(100vh - 60px)",
        }}
      >
        {/* Sticky progress sidebar */}
        <div
          style={{
            position: "fixed",
            right: "2rem",
            top: "50%",
            transform: "translateY(-50%)",
            display: "flex",
            flexDirection: "column",
            gap: "0.4rem",
            zIndex: 50,
          }}
        >
          {SECTIONS.map((s, i) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              title={s.title}
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: i === activeIdx ? "var(--teal-light)" : "rgba(148,163,184,0.25)",
                transition: "all 0.25s ease",
                display: "block",
                transform: i === activeIdx ? "scale(1.4)" : "scale(1)",
              }}
            />
          ))}
        </div>

        {/* ── Section 1: The Problem ──────────────────────────────────────── */}
        <section
          ref={refs[0] as React.RefObject<HTMLElement>}
          id="problem"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start", borderBottom: "1px solid rgba(148,163,184,0.06)" }}
        >
          <div style={{ maxWidth: "800px", margin: "0 auto", width: "100%", opacity: visibles[0] ? 1 : 0.3, transform: visibles[0] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[0] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              01 / The Problem
            </div>
            <h1 style={{ fontSize: "3rem", fontWeight: 900, lineHeight: 1.15, color: "var(--text-primary)", marginBottom: "1.5rem" }}>
              Most cancer patients never get a{" "}
              <span style={{ color: "var(--teal-light)" }}>tumour board</span>.
            </h1>
            <p style={{ fontSize: "1.05rem", color: "var(--text-secondary)", lineHeight: 1.8, maxWidth: "620px", marginBottom: "2rem" }}>
              A multidisciplinary tumour board — where pathologists, researchers, and oncologists deliberate together — is the gold standard for complex cancer cases. But they take weeks to convene, require specialist co-location, and are unavailable in most of the world.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
              {[
                { stat: "2–4 weeks", desc: "Average time to convene a tumour board" },
                { stat: "<20%", desc: "Of cancer patients globally access one" },
                { stat: "35%", desc: "Of diagnoses changed after board review" },
              ].map(({ stat, desc }) => (
                <div key={stat} className="glass-card-sm" style={{ padding: "1.1rem" }}>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--teal-light)", marginBottom: "0.3rem" }}>{stat}</div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Section 2: Three Agents Enter ──────────────────────────────── */}
        <section
          ref={refs[1] as React.RefObject<HTMLElement>}
          id="team"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start", borderBottom: "1px solid rgba(148,163,184,0.06)" }}
        >
          <div style={{ maxWidth: "900px", margin: "0 auto", width: "100%", opacity: visibles[1] ? 1 : 0.3, transform: visibles[1] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[1] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              02 / Three Agents Enter the Room
            </div>
            <h2 style={{ fontSize: "2.2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "1.5rem", lineHeight: 1.2 }}>
              A digital board that reasons, debates, and decides.
            </h2>
            <div style={{ display: "flex", justifyContent: "center", gap: "3rem", flexWrap: "wrap", marginTop: "2rem" }}>
              {(["pathologist", "researcher", "oncologist"] as PersonaId[]).map((id, i) => (
                <AgentPersona
                  key={id}
                  id={id}
                  state={visibles[1] ? "active" : "idle"}
                  size={100}
                  showLabel
                />
              ))}
            </div>
            <p style={{ fontSize: "0.95rem", color: "var(--text-secondary)", lineHeight: 1.75, maxWidth: "640px", margin: "2rem auto 0", textAlign: "center" }}>
              Five specialist AI models collaborate sequentially — vision, language-vision, retrieval, staging, and synthesis. Each model runs simultaneously in the MI300X&apos;s 192 GB unified memory.
            </p>
          </div>
        </section>

        {/* ── Section 3: Pathologist Sees ────────────────────────────────── */}
        <section
          ref={refs[2] as React.RefObject<HTMLElement>}
          id="pathologist"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start", borderBottom: "1px solid rgba(148,163,184,0.06)" }}
        >
          <div style={{ maxWidth: "900px", margin: "0 auto", width: "100%", opacity: visibles[2] ? 1 : 0.3, transform: visibles[2] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[2] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              03 / The Pathologist Sees Pixels
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "2.5rem", alignItems: "center" }}>
              <AgentPersona id="pathologist" state={visibles[2] ? "speaking" : "idle"} size={110} showLabel />
              <div>
                <h2 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "1rem", lineHeight: 1.2 }}>
                  GigaPath sees what the human eye misses.
                </h2>
                <p style={{ fontSize: "0.92rem", color: "var(--text-secondary)", lineHeight: 1.75, marginBottom: "1.25rem" }}>
                  Prov-GigaPath (1.1B parameters, trained on 1.3 billion pathology tokens) encodes every 224×224 patch into a rich morphological embedding. Attention rollout reveals which regions drive the diagnosis.
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {[
                    "🔬 Nuclear atypia detected in 3 of 12 patches",
                    "📊 MC Dropout uncertainty: 91% ± 4.2%",
                    "🧬 EGFR mutation morphology pattern: high probability",
                  ].map((line) => (
                    <div key={line} style={{ fontSize: "0.83rem", color: "var(--text-secondary)", display: "flex", gap: "0.4rem" }}>
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Section 4: Researcher Cites ────────────────────────────────── */}
        <section
          ref={refs[3] as React.RefObject<HTMLElement>}
          id="researcher"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start", borderBottom: "1px solid rgba(148,163,184,0.06)" }}
        >
          <div style={{ maxWidth: "900px", margin: "0 auto", width: "100%", opacity: visibles[3] ? 1 : 0.3, transform: visibles[3] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[3] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              04 / The Researcher Cites NCCN
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "2.5rem", alignItems: "center" }}>
              <div>
                <h2 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "1rem", lineHeight: 1.2 }}>
                  Every recommendation is cited.
                </h2>
                <p style={{ fontSize: "0.92rem", color: "var(--text-secondary)", lineHeight: 1.75, marginBottom: "1.25rem" }}>
                  RAG over pre-indexed NCCN guidelines, TCGA studies, and PubMed abstracts. The researcher retrieves evidence, reranks it, and challenges the oncologist if the initial plan misses a guideline.
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  {[
                    { title: "NCCN NSCLC Guidelines 2024", category: "Category 1", color: "#0d9488" },
                    { title: "FLAURA Trial — Osimertinib", category: "Phase III RCT", color: "#22c55e" },
                    { title: "EGFR Mutation Testing Recommendations", category: "CAP/IASLC", color: "#38bdf8" },
                  ].map(({ title, category, color }) => (
                    <div key={title} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.6rem 0.9rem", background: `${color}08`, border: `1px solid ${color}30`, borderRadius: "8px" }}>
                      <div style={{ width: "3px", height: "28px", background: color, borderRadius: "2px", flexShrink: 0 }} />
                      <div>
                        <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-primary)" }}>{title}</div>
                        <div style={{ fontSize: "0.7rem", color, marginTop: "0.1rem" }}>{category}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <AgentPersona id="researcher" state={visibles[3] ? "speaking" : "idle"} size={110} showLabel />
            </div>
          </div>
        </section>

        {/* ── Section 5: The Debate ───────────────────────────────────────── */}
        <section
          ref={refs[4] as React.RefObject<HTMLElement>}
          id="debate"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start", borderBottom: "1px solid rgba(148,163,184,0.06)" }}
        >
          <div style={{ maxWidth: "900px", margin: "0 auto", width: "100%", opacity: visibles[4] ? 1 : 0.3, transform: visibles[4] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[4] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              05 / The Debate — Agents Change Each Other&apos;s Minds
            </div>
            <h2 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "1.5rem", lineHeight: 1.2 }}>
              The researcher caught a missing EGFR test.
            </h2>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
              {/* Challenge bubble */}
              <div style={{ padding: "1.1rem", background: "rgba(217,119,6,0.06)", border: "1px solid rgba(217,119,6,0.25)", borderRadius: "10px" }}>
                <div style={{ fontSize: "0.7rem", color: "#d97706", fontWeight: 700, marginBottom: "0.5rem" }}>📚 RESEARCHER CHALLENGE</div>
                <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
                  ⚠️ EGFR status unknown. NCCN Category 1 for first-line TKI only applies to EGFR-mutant. Recommend molecular testing before committing to chemotherapy.
                </p>
              </div>
              {/* Revision bubble */}
              <div style={{ padding: "1.1rem", background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.2)", borderRadius: "10px" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--success)", fontWeight: 700, marginBottom: "0.5rem" }}>👨‍⚕️ ONCOLOGIST REVISION</div>
                <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
                  Acknowledged. Revising first-line pending EGFR result. Adding osimertinib as conditional Category 1 recommendation.
                </p>
              </div>
            </div>

            {/* Live revision diff */}
            <RevisionDiff
              before={beforeRevision}
              after={afterRevision}
              label="Revision diff — what changed after debate"
            />

            <div style={{ marginTop: "1.25rem", fontSize: "0.8rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ padding: "0.2rem 0.6rem", borderRadius: "20px", background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.3)", color: "var(--success)", fontSize: "0.72rem", fontWeight: 700 }}>
                Consensus Score: 87/100
              </span>
              This is what distinguishes AOB from a single-LLM answer.
            </div>
          </div>
        </section>

        {/* ── Section 6: Hardware Proof ───────────────────────────────────── */}
        <section
          ref={refs[5] as React.RefObject<HTMLElement>}
          id="hardware"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start", borderBottom: "1px solid rgba(148,163,184,0.06)" }}
        >
          <div style={{ maxWidth: "1000px", margin: "0 auto", width: "100%", opacity: visibles[5] ? 1 : 0.3, transform: visibles[5] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[5] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              06 / The Math Behind the Architecture
            </div>
            <h2 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "1rem", lineHeight: 1.2 }}>
              This architecture is{" "}
              <span style={{ color: "#f87171", textDecoration: "line-through" }}>impossible</span>{" "}
              <span style={{ color: "var(--teal-light)" }}>on a single H100</span>.
            </h2>
            <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: "660px", marginBottom: "1.75rem" }}>
              Toggle components to see the memory arithmetic. The MI300X&apos;s 192 GB HBM3 unified pool is not a performance advantage — it&apos;s what makes this architecture physically possible.
            </p>
            <H100Simulator />
          </div>
        </section>

        {/* ── Section 7: The Benchmark ───────────────────────────────────── */}
        <section
          ref={refs[6] as React.RefObject<HTMLElement>}
          id="benchmark"
          style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "6rem 2rem", scrollSnapAlign: "start" }}
        >
          <div style={{ maxWidth: "1000px", margin: "0 auto", width: "100%", opacity: visibles[6] ? 1 : 0.3, transform: visibles[6] ? "translateY(0)" : "translateY(30px)", transition: "all 0.7s ease" }}>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, color: visibles[6] ? "var(--teal-light)" : "var(--text-muted)", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              07 / The Benchmark Proves It
            </div>
            <h2 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "0.75rem", lineHeight: 1.2 }}>
              82.3% TNM accuracy. 74.8% biomarker F1. Reproducible.
            </h2>
            <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: "640px", marginBottom: "1.75rem" }}>
              Evaluated on 100 curated clinical cases. All metrics with 95% bootstrap CIs. Dataset published on HuggingFace for independent verification.
            </p>
            <div className="glass-card" style={{ padding: "1.5rem" }}>
              <Leaderboard data={FALLBACK_ABLATION_INLINE} />
            </div>
            <div style={{ marginTop: "1.5rem", display: "flex", gap: "1rem", flexWrap: "wrap" }}>
              <a href="/benchmark" className="btn-teal" style={{ padding: "0.6rem 1.5rem", textDecoration: "none", fontSize: "0.9rem" }}>
                Full Benchmark →
              </a>
              <a href="/" className="btn-ghost" style={{ padding: "0.6rem 1.5rem", textDecoration: "none", fontSize: "0.9rem" }}>
                Run a Case →
              </a>
            </div>
          </div>
        </section>

      </div>
    </>
  );
}
