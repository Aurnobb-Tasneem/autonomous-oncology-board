"use client";

/**
 * Word-level LCS diff renderer.
 * No new dependencies — uses pure JS and existing .diff-removed / .diff-added
 * CSS classes from globals.css (lines 176–187).
 *
 * Usage:
 *   <RevisionDiff before="old treatment text" after="revised treatment text" />
 */

interface DiffToken {
  type: "equal" | "removed" | "added";
  text: string;
}

/**
 * Compute word-level LCS diff between two strings.
 * Returns a flat list of tokens annotated with equal/removed/added.
 */
function diffWords(before: string, after: string): DiffToken[] {
  // Split preserving whitespace chunks so the rendered output is readable.
  const bTokens = before.split(/(\s+)/);
  const aTokens = after.split(/(\s+)/);

  const m = bTokens.length;
  const n = aTokens.length;

  // LCS DP table (space-optimised: two rows).
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        bTokens[i - 1] === aTokens[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  // Backtrack to produce diff tokens.
  const result: DiffToken[] = [];
  let i = m;
  let j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && bTokens[i - 1] === aTokens[j - 1]) {
      result.unshift({ type: "equal", text: bTokens[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.unshift({ type: "added", text: aTokens[j - 1] });
      j--;
    } else {
      result.unshift({ type: "removed", text: bTokens[i - 1] });
      i--;
    }
  }

  // Collapse consecutive same-type tokens (except equal whitespace-only tokens)
  // to reduce DOM node count.
  const collapsed: DiffToken[] = [];
  for (const tok of result) {
    const prev = collapsed[collapsed.length - 1];
    if (prev && prev.type === tok.type && tok.type !== "equal") {
      prev.text += tok.text;
    } else {
      collapsed.push({ ...tok });
    }
  }

  return collapsed;
}

// ---------------------------------------------------------------------------

interface RevisionDiffProps {
  before: string;
  after: string;
  /** Optional label shown above the diff. Defaults to "Revision". */
  label?: string;
  /** Whether to show the full before/after strings as collapsed toggle. Default false. */
  showRaw?: boolean;
}

export default function RevisionDiff({
  before,
  after,
  label = "Revision",
  showRaw = false,
}: RevisionDiffProps) {
  if (!before && !after) return null;
  if (!before) {
    return (
      <div className="diff-block">
        <span className="diff-added">{after}</span>
      </div>
    );
  }
  if (!after) {
    return (
      <div className="diff-block">
        <span className="diff-removed">{before}</span>
      </div>
    );
  }

  const tokens = diffWords(before, after);
  const hasChanges = tokens.some((t) => t.type !== "equal");

  if (!hasChanges) {
    return (
      <div
        style={{
          fontSize: "0.78rem",
          color: "var(--text-muted)",
          fontStyle: "italic",
        }}
      >
        No changes in this revision.
      </div>
    );
  }

  return (
    <div
      style={{
        borderRadius: "8px",
        overflow: "hidden",
        border: "1px solid rgba(148,163,184,0.14)",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "0.35rem 0.75rem",
          background: "rgba(13,148,136,0.08)",
          borderBottom: "1px solid rgba(148,163,184,0.1)",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          fontSize: "0.7rem",
          fontWeight: 700,
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        <span style={{ color: "#f87171" }}>−</span>
        <span style={{ color: "#4ade80" }}>+</span>
        {label}
      </div>

      {/* Inline word diff */}
      <div
        style={{
          padding: "0.75rem 0.9rem",
          fontSize: "0.85rem",
          lineHeight: 1.7,
          background: "rgba(5,10,25,0.6)",
          wordBreak: "break-word",
        }}
      >
        {tokens.map((tok, i) => {
          if (tok.type === "equal") {
            return <span key={i}>{tok.text}</span>;
          }
          if (tok.type === "removed") {
            return (
              <span
                key={i}
                className="diff-removed"
                style={{
                  background: "rgba(239,68,68,0.15)",
                  color: "#fca5a5",
                  textDecoration: "line-through",
                  borderRadius: "3px",
                  padding: "0 2px",
                }}
              >
                {tok.text}
              </span>
            );
          }
          return (
            <span
              key={i}
              className="diff-added"
              style={{
                background: "rgba(74,222,128,0.14)",
                color: "#86efac",
                borderRadius: "3px",
                padding: "0 2px",
                fontWeight: 600,
              }}
            >
              {tok.text}
            </span>
          );
        })}
      </div>
    </div>
  );
}
