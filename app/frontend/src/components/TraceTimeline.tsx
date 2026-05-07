import { useState, useEffect } from "react";
import type { CSSProperties } from "react";
import type { TracePayload } from "../hooks/useSSE";
import styles from "./TraceTimeline.module.css";

interface TraceRow {
  step: string;
  status: "running" | "done" | "error";
  info: string;
}

function buildRows(traces: TracePayload[]): TraceRow[] {
  const rows: TraceRow[] = [];
  for (const t of traces) {
    if (t.status === "start") {
      rows.push({ step: t.step, status: "running", info: t.info });
    } else if (t.status === "done" || t.status === "error") {
      const idx = [...rows].reverse().findIndex((r) => r.step === t.step && r.status === "running");
      if (idx !== -1) {
        rows[rows.length - 1 - idx] = { step: t.step, status: t.status === "done" ? "done" : "error", info: t.info };
      } else {
        rows.push({ step: t.step, status: t.status === "done" ? "done" : "error", info: t.info });
      }
    }
  }
  const last = rows[rows.length - 1];
  const synthDoneCount = rows.filter((r) => r.step === "synthesize" && r.status === "done").length;
  const critiqueCount  = rows.filter((r) => r.step === "critique").length;
  if (last?.step === "synthesize" && last.status === "done" && critiqueCount < synthDoneCount) {
    rows.push({ step: "critique", status: "running", info: "" });
  } else if (last?.step === "critique" && last.status === "done" && last.info === "FAIL") {
    const retrieveDoneCount = rows.filter((r) => r.step === "retrieve" && r.status === "done").length;
    if (retrieveDoneCount === 1) {
      rows.push({ step: "decompose", status: "running", info: "" });
    } else if (!rows.some((r) => r.step === "correct")) {
      rows.push({ step: "correct", status: "running", info: "" });
    }
  }
  return rows;
}

function dotStyle(status: TraceRow["status"], info: string): CSSProperties {
  const isFail = status === "error" || (status === "done" && info === "FAIL");
  const isPass = status === "done" && info === "PASS";
  if (isFail) return { background: "oklch(0.72 0.16 20 / 0.2)", border: "1px solid oklch(0.72 0.16 20)" };
  if (isPass) return { background: "oklch(0.78 0.14 155 / 0.2)", border: "1px solid oklch(0.78 0.14 155)" };
  if (status === "running") return { background: "var(--accent-soft)", border: "1px solid var(--accent)", animation: "breathe 1.6s ease-in-out infinite" };
  return { background: "var(--accent-soft)", border: "1px solid var(--accent)" };
}

function metaColor(status: TraceRow["status"], info: string): string {
  if (status === "error" || (status === "done" && info === "FAIL")) return "oklch(0.82 0.14 20)";
  if (status === "done" && info === "PASS") return "oklch(0.85 0.14 155)";
  return "var(--fg-3)";
}

interface Props {
  traces: TracePayload[];
  isStreaming: boolean;
}

export default function TraceTimeline({ traces, isStreaming }: Props) {
  const rows = buildRows(traces);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (isStreaming) setCollapsed(false);
  }, [isStreaming]);

  if (rows.length === 0) return null;

  const doneCount = rows.filter((r) => r.status !== "running").length;

  return (
    <div className={styles.card}>
      <button
        onClick={() => setCollapsed((c) => !c)}
        className={styles.toggle}
        data-expanded={!collapsed}
      >
        <span className={`mono ${styles.label}`}>
          {isStreaming
            ? `Reasoning · ${rows.length} step${rows.length !== 1 ? "s" : ""}…`
            : `Reasoning · ${rows.length} steps`}
        </span>
        <span className={styles.spacer} />
        <svg
          width={12} height={12} viewBox="0 0 24 24" fill="none"
          stroke="var(--fg-4)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"
          className={styles.chevron}
          data-collapsed={collapsed}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {!collapsed && (
        <div className={styles.rows}>
          <div className={styles.rail} />
          {rows.map((row, i) => {
            const infoLines = row.info ? row.info.split("\n").filter(Boolean) : [];
            const isMultiLine = infoLines.length > 1;
            const metaText = !isMultiLine && infoLines[0] ? infoLines[0] : null;
            return (
              <div key={i} className={styles.row}>
                <div className={styles.dot} style={dotStyle(row.status, row.info)} />
                <div className={styles.rowContent}>
                  <div className={styles.rowHeader}>
                    <span className={`mono ${styles.stepName}`}>{row.step}</span>
                    {metaText && (
                      <span className={`mono ${styles.stepMeta}`} style={{ color: metaColor(row.status, row.info) }}>{metaText}</span>
                    )}
                  </div>
                  {isMultiLine && (
                    <div className={styles.infoLines}>
                      {infoLines.map((line, j) => (
                        <div key={j} className={`mono ${styles.infoLine}`}>{line}</div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {collapsed && doneCount > 0 && (
        <div className={`mono ${styles.summary}`}>
          {doneCount} of {rows.length} steps complete
        </div>
      )}
    </div>
  );
}
