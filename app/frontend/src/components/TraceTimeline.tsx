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

  // Auto-expand while streaming, auto-collapse when done
  useEffect(() => {
    if (isStreaming) setCollapsed(false);
    else setCollapsed(true);
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
          {rows.map((row, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="mt-px flex-shrink-0 w-4 flex justify-center">
                <StatusIcon status={row.status} />
              </span>
              <span className="font-mono text-gray-700">{row.step}</span>
              {row.info && (
                <span className="text-gray-400 truncate max-w-[240px]">{row.info}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
