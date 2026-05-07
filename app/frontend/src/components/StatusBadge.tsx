import type { Document } from "../api";

export default function StatusBadge({ status }: { status: Document["status"] }) {
  if (status === "ready") {
    return (
      <span className="pill ready">
        <svg width={9} height={9} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
          <path d="m4 12 5 5L20 6" />
        </svg>
        ready
      </span>
    );
  }
  if (status === "ingesting") {
    return (
      <span className="pill ingest">
        <span className="dot breathe" style={{ background: "var(--amber)" }} />
        ingesting
      </span>
    );
  }
  return <span className="pill failed">failed</span>;
}
