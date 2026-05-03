import type { Document } from "../api";

const styles: Record<Document["status"], string> = {
  ingesting: "bg-yellow-100 text-yellow-700",
  ready:     "bg-green-100 text-green-700",
  failed:    "bg-red-100 text-red-700",
};

export default function StatusBadge({ status }: { status: Document["status"] }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${styles[status]}`}>
      {status === "ingesting" && (
        <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" d="M12 3v3m0 12v3M3 12h3m12 0h3" />
        </svg>
      )}
      {status}
    </span>
  );
}
