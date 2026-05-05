import { useState, useRef, useEffect } from "react";
import type { Collection } from "../api";

interface Props {
  collections: Collection[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
  onCreate: (name: string) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  onDocDrop: (docId: string, collectionId: string) => void;
  totalDocs: number;
}

export default function CollectionSidebar({
  collections,
  activeId,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onDocDrop,
  totalDocs,
}: Props) {
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [dropTargetId, setDropTargetId] = useState<string | null>(null);
  const newInputRef = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const dragCounters = useRef<Record<string, number>>({});

  useEffect(() => {
    if (creating) newInputRef.current?.focus();
  }, [creating]);

  useEffect(() => {
    if (renamingId) renameInputRef.current?.focus();
  }, [renamingId]);

  function submitCreate() {
    const name = newName.trim();
    if (name) onCreate(name);
    setCreating(false);
    setNewName("");
  }

  function submitRename(id: string) {
    const name = renameValue.trim();
    if (name) onRename(id, name);
    setRenamingId(null);
    setRenameValue("");
  }

  function handleDragEnter(e: React.DragEvent, collectionId: string) {
    e.preventDefault();
    dragCounters.current[collectionId] = (dragCounters.current[collectionId] ?? 0) + 1;
    if (dragCounters.current[collectionId] === 1) setDropTargetId(collectionId);
  }

  function handleDragLeave(e: React.DragEvent, collectionId: string) {
    e.preventDefault();
    dragCounters.current[collectionId] = Math.max(0, (dragCounters.current[collectionId] ?? 0) - 1);
    if (dragCounters.current[collectionId] === 0) setDropTargetId(null);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }

  function handleDrop(e: React.DragEvent, collectionId: string) {
    e.preventDefault();
    dragCounters.current[collectionId] = 0;
    setDropTargetId(null);
    const docId = e.dataTransfer.getData("application/x-doc-id");
    if (docId) onDocDrop(docId, collectionId);
  }

  return (
    <aside className="w-52 shrink-0 border-r border-gray-200 flex flex-col bg-gray-50 overflow-y-auto">
      <div className="px-3 pt-5 pb-2">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Collections
        </span>
      </div>

      {/* All Documents */}
      <button
        onClick={() => onSelect(null)}
        className={`flex items-center justify-between mx-2 px-2 py-1.5 rounded-lg text-sm font-medium transition-colors ${
          activeId === null
            ? "bg-indigo-100 text-indigo-700"
            : "text-gray-600 hover:bg-gray-200"
        }`}
      >
        <span>All Documents</span>
        <span className="text-xs text-gray-400">{totalDocs}</span>
      </button>

      {/* Collection list */}
      <div className="mt-1 flex flex-col gap-0.5 px-2">
        {collections.map((c) => (
          <div
            key={c.id}
            className="group relative"
            onDragEnter={(e) => handleDragEnter(e, c.id)}
            onDragLeave={(e) => handleDragLeave(e, c.id)}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, c.id)}
          >
            {renamingId === c.id ? (
              <input
                ref={renameInputRef}
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={() => submitRename(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submitRename(c.id);
                  if (e.key === "Escape") { setRenamingId(null); }
                }}
                className="w-full px-2 py-1.5 text-sm rounded-lg border border-indigo-400 outline-none bg-white"
              />
            ) : (
              <button
                onClick={() => onSelect(c.id)}
                onDoubleClick={() => { setRenamingId(c.id); setRenameValue(c.name); }}
                className={`w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-sm transition-colors text-left ${
                  dropTargetId === c.id
                    ? "bg-indigo-100 ring-2 ring-indigo-400 text-indigo-700"
                    : activeId === c.id
                    ? "bg-indigo-100 text-indigo-700"
                    : "text-gray-600 hover:bg-gray-200"
                }`}
              >
                <span className="truncate">{c.name}</span>
                {dropTargetId === c.id ? (
                  <svg className="w-3.5 h-3.5 text-indigo-500 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                ) : (
                  <span className="text-xs text-gray-400 shrink-0 ml-1">{c.doc_count}</span>
                )}
              </button>
            )}

            {renamingId !== c.id && dropTargetId !== c.id && (
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
                title="Delete collection"
                className="absolute right-1 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center justify-center w-5 h-5 rounded text-gray-400 hover:text-red-500 hover:bg-red-50"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Create new */}
      <div className="px-2 mt-2">
        {creating ? (
          <input
            ref={newInputRef}
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onBlur={submitCreate}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitCreate();
              if (e.key === "Escape") { setCreating(false); setNewName(""); }
            }}
            placeholder="Collection name…"
            className="w-full px-2 py-1.5 text-sm rounded-lg border border-indigo-400 outline-none bg-white"
          />
        ) : (
          <button
            onClick={() => setCreating(true)}
            className="flex items-center gap-1 px-2 py-1.5 text-sm text-gray-400 hover:text-indigo-600 rounded-lg hover:bg-gray-200 w-full transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            New collection
          </button>
        )}
      </div>
    </aside>
  );
}
