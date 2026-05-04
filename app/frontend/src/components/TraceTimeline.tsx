import { useState, useEffect } from "react";
import type { TracePayload } from "../hooks/useSSE";

// ---------------------------------------------------------------------------
// Collapse each start/done pair into a single display row
// ---------------------------------------------------------------------------

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
      // Find the last running row for this step and promote it
      const idx = [...rows].reverse().findIndex((r) => r.step === t.step && r.status === "running");
      if (idx !== -1) {
        const realIdx = rows.length - 1 - idx;
        rows[realIdx] = { step: t.step, status: t.status === "done" ? "done" : "error", info: t.info };
      } else {
        rows.push({ step: t.step, status: t.status === "done" ? "done" : "error", info: t.info });
      }
    }
  }

  // Infer the next expected step so spinners appear even when start+done
  // arrive batched in the same TCP chunk.
  const last = rows[rows.length - 1];
  const synthDoneCount = rows.filter((r) => r.step === "synthesize" && r.status === "done").length;
  const critiqueCount = rows.filter((r) => r.step === "critique").length;

  if (last?.step === "synthesize" && last.status === "done") {
    // Each synthesize is followed by a critique; infer it if not yet visible.
    if (critiqueCount < synthDoneCount) {
      rows.push({ step: "critique", status: "running", info: "" });
    }
  } else if (last?.step === "critique" && last.status === "done" && last.info === "FAIL") {
    const retrieveDoneCount = rows.filter((r) => r.step === "retrieve" && r.status === "done").length;
    if (retrieveDoneCount === 1) {
      // Hop 1 failed — hop 2 decompose is imminent.
      rows.push({ step: "decompose", status: "running", info: "" });
    } else if (!rows.some((r) => r.step === "correct")) {
      // Hop 2 failed — inferential correction step is imminent.
      rows.push({ step: "correct", status: "running", info: "" });
    }
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function StatusIcon({ status }: { status: TraceRow["status"] }) {
  if (status === "running") {
    return (
      <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-gray-400 border-t-transparent animate-spin" />
    );
  }
  if (status === "done") {
    return <span className="text-green-500 text-xs font-bold leading-none">✓</span>;
  }
  return <span className="text-red-500 text-xs font-bold leading-none">✕</span>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  traces: TracePayload[];
  isStreaming: boolean;
}

export default function TraceTimeline({ traces, isStreaming }: Props) {
  const rows = buildRows(traces);
  const [collapsed, setCollapsed] = useState(false);

  // Expand when new streaming starts
  useEffect(() => {
    if (isStreaming) setCollapsed(false);
  }, [isStreaming]);

  if (rows.length === 0) return null;

  return (
    <div className="border-b border-gray-100 text-xs text-gray-500">
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center gap-1.5 px-4 py-2 hover:bg-gray-50 transition-colors text-left"
      >
        <span className={`transition-transform duration-150 ${collapsed ? "" : "rotate-90"}`}>
          ▶
        </span>
        <span className="font-medium text-gray-600">
          {isStreaming ? "Thinking…" : `${rows.length} step${rows.length !== 1 ? "s" : ""}`}
        </span>
      </button>

      {!collapsed && (
        <ul className="px-4 pb-2 space-y-1.5">
          {rows.map((row, i) => {
            const infoLines = row.info ? row.info.split("\n").filter(Boolean) : [];
            const multiLine = infoLines.length > 1;
            return (
            <li key={i} className="flex items-start gap-2">
              <span className="mt-px flex-shrink-0 w-4 flex justify-center">
                <StatusIcon status={row.status} />
              </span>
              <div className="min-w-0">
                <span className="font-mono text-gray-700">{row.step}</span>
                {!multiLine && infoLines[0] && (
                  <span className="ml-2 text-gray-400">{infoLines[0]}</span>
                )}
                {multiLine && (
                  <ul className="mt-1 space-y-0.5 text-gray-400">
                    {infoLines.map((line, j) => (
                      <li key={j} className="flex gap-1"><span>–</span><span>{line}</span></li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
