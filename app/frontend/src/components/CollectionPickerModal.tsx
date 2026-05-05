import { useState } from "react";
import type { Collection, Document } from "../api";

export default function CollectionPickerModal({
  doc,
  collections,
  onConfirm,
  onSkip,
}: {
  doc: Document;
  collections: Collection[];
  onConfirm: (collectionIds: string[]) => void;
  onSkip: () => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={onSkip} />
      <div className="relative bg-white rounded-2xl shadow-xl w-full max-w-sm mx-4 p-6 flex flex-col gap-5">
        <div>
          <h2 className="text-base font-semibold text-gray-900">Add to collection</h2>
          <p className="mt-0.5 text-sm text-gray-500 truncate">
            {doc.title}
          </p>
        </div>

        {collections.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-4">
            No collections yet. Create one in the sidebar first.
          </p>
        ) : (
          <ul className="flex flex-col gap-1 max-h-60 overflow-y-auto -mx-1">
            {collections.map((c) => (
              <li key={c.id}>
                <label className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-50 cursor-pointer">
                  <span
                    className={`w-4.5 h-4.5 rounded border flex items-center justify-center shrink-0 transition-colors ${
                      selected.has(c.id)
                        ? "bg-indigo-600 border-indigo-600"
                        : "border-gray-300"
                    }`}
                    style={{ width: 18, height: 18 }}
                  >
                    {selected.has(c.id) && (
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" strokeWidth={3} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                      </svg>
                    )}
                  </span>
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={selected.has(c.id)}
                    onChange={() => toggle(c.id)}
                  />
                  <span className="text-sm text-gray-700">{c.name}</span>
                  <span className="ml-auto text-xs text-gray-400">{c.doc_count}</span>
                </label>
              </li>
            ))}
          </ul>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button
            onClick={onSkip}
            className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Skip
          </button>
          <button
            onClick={() => onConfirm(Array.from(selected))}
            disabled={selected.size === 0}
            className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Add to {selected.size > 1 ? `${selected.size} collections` : "collection"}
          </button>
        </div>
      </div>
    </div>
  );
}
