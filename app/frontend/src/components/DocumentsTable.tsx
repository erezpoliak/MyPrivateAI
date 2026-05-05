import { useState, useRef, useEffect } from "react";
import type { Collection, Document } from "../api";
import StatusBadge from "./StatusBadge";

function AssignPopover({
  doc,
  collections,
  onAssign,
  onUnassign,
  onClose,
}: {
  doc: Document;
  collections: Collection[];
  onAssign: (docId: string, collectionId: string) => void;
  onUnassign: (docId: string, collectionId: string) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  if (collections.length === 0) {
    return (
      <div ref={ref} className="absolute z-20 right-0 top-full mt-1 w-44 bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-xs text-gray-400">
        No collections yet. Create one in the sidebar.
      </div>
    );
  }

  return (
    <div ref={ref} className="absolute z-20 right-0 top-full mt-1 w-52 bg-white border border-gray-200 rounded-lg shadow-lg py-1">
      {collections.map((c) => {
        const assigned = doc.collection_ids.includes(c.id);
        return (
          <button
            key={c.id}
            onClick={() => assigned ? onUnassign(doc.id, c.id) : onAssign(doc.id, c.id)}
            className="flex items-center gap-2 w-full px-3 py-1.5 text-sm text-left hover:bg-gray-50 transition-colors"
          >
            <span className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 ${assigned ? "bg-indigo-600 border-indigo-600" : "border-gray-300"}`}>
              {assigned && (
                <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" strokeWidth={3} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                </svg>
              )}
            </span>
            <span className="truncate text-gray-700">{c.name}</span>
          </button>
        );
      })}
    </div>
  );
}

export default function DocumentsTable({
  docs,
  collections,
  onDelete,
  onAssign,
  onUnassign,
}: {
  docs: Document[];
  collections: Collection[];
  onDelete: (id: string) => void;
  onAssign: (docId: string, collectionId: string) => void;
  onUnassign: (docId: string, collectionId: string) => void;
}) {
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [assigningId, setAssigningId] = useState<string | null>(null);

  const collectionMap = Object.fromEntries(collections.map((c) => [c.id, c.name]));

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
            <th className="px-4 py-3">Collections</th>
            <th className="px-4 py-3 text-right">Pages</th>
            <th className="px-4 py-3 text-right">Chunks</th>
            <th className="px-4 py-3">Uploaded</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {docs.map((doc) => (
            <tr
              key={doc.id}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData("application/x-doc-id", doc.id);
                e.dataTransfer.effectAllowed = "copy";
              }}
              className="hover:bg-gray-50 transition-colors cursor-grab active:cursor-grabbing"
            >
              <td className="px-4 py-3 font-medium text-gray-800 max-w-[160px] truncate">
                {doc.title}
              </td>
              <td className="px-4 py-3">
                <div className="relative flex items-center gap-1 flex-wrap">
                  {doc.collection_ids.map((cid) =>
                    collectionMap[cid] ? (
                      <span
                        key={cid}
                        className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-100"
                      >
                        {collectionMap[cid]}
                      </span>
                    ) : null
                  )}
                  <button
                    onClick={() => setAssigningId(assigningId === doc.id ? null : doc.id)}
                    title="Assign to collection"
                    className="inline-flex items-center justify-center w-5 h-5 rounded border border-dashed border-gray-300 text-gray-400 hover:border-indigo-400 hover:text-indigo-500 transition-colors"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                    </svg>
                  </button>
                  {assigningId === doc.id && (
                    <AssignPopover
                      doc={doc}
                      collections={collections}
                      onAssign={(docId, cid) => { onAssign(docId, cid); }}
                      onUnassign={(docId, cid) => { onUnassign(docId, cid); }}
                      onClose={() => setAssigningId(null)}
                    />
                  )}
                </div>
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
