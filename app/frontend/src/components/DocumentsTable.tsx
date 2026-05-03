import { useState } from "react";
import type { Document } from "../api";
import StatusBadge from "./StatusBadge";

export default function DocumentsTable({
  docs,
  onDelete,
}: {
  docs: Document[];
  onDelete: (id: string) => void;
}) {
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  if (docs.length === 0) {
    return (
      <p className="text-sm text-gray-400 text-center py-8">
        No documents yet — upload a PDF above.
      </p>
    );
  }

  return (
    <div className="mt-8 overflow-x-auto rounded-xl border border-gray-200">
      <table className="w-full text-sm text-left">
        <thead className="bg-gray-50 text-gray-500 uppercase text-xs tracking-wide">
          <tr>
            <th className="px-4 py-3">Title</th>
            <th className="px-4 py-3">Filename</th>
            <th className="px-4 py-3 text-right">Pages</th>
            <th className="px-4 py-3 text-right">Chunks</th>
            <th className="px-4 py-3">Uploaded</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {docs.map((doc) => (
            <tr key={doc.id} className="hover:bg-gray-50 transition-colors">
              <td className="px-4 py-3 font-medium text-gray-800 max-w-[180px] truncate">
                {doc.title}
              </td>
              <td className="px-4 py-3 text-gray-500 max-w-[160px] truncate">
                {doc.filename}
              </td>
              <td className="px-4 py-3 text-right text-gray-600">
                {doc.page_count || "—"}
              </td>
              <td className="px-4 py-3 text-right text-gray-600">
                {doc.chunk_count || "—"}
              </td>
              <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                {new Date(doc.created_at).toLocaleDateString()}
              </td>
              <td className="px-4 py-3">
                <StatusBadge status={doc.status} />
              </td>
              <td className="px-4 py-3 text-right whitespace-nowrap">
                {confirmingId === doc.id ? (
                  <span className="inline-flex items-center gap-2">
                    <span className="text-xs text-gray-500">Delete?</span>
                    <button
                      onClick={() => { onDelete(doc.id); setConfirmingId(null); }}
                      className="text-xs font-medium text-red-600 hover:text-red-700"
                    >
                      Yes
                    </button>
                    <button
                      onClick={() => setConfirmingId(null)}
                      className="text-xs font-medium text-gray-500 hover:text-gray-700"
                    >
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    onClick={() => setConfirmingId(doc.id)}
                    className="text-gray-400 hover:text-red-500 transition-colors"
                    title="Delete document"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                    </svg>
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
