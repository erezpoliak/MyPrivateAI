import { useState, useRef, useEffect } from "react";
import type { Collection } from "../api";
import styles from "./CollectionSidebar.module.css";

interface Props {
  collections: Collection[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
  onCreate: (name: string) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  totalDocs: number;
  collectionDots: string[];
}

export default function CollectionSidebar({
  collections, activeId, onSelect, onCreate, onRename, onDelete, totalDocs, collectionDots,
}: Props) {
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const newInputRef = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { if (creating) newInputRef.current?.focus(); }, [creating]);
  useEffect(() => { if (renamingId) renameInputRef.current?.focus(); }, [renamingId]);

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

  return (
    <aside className={styles.sidebar}>
      <span className={`mono ${styles.sectionLabel}`}>Collections</span>

      <button
        onClick={() => onSelect(null)}
        className={styles.allDocs}
        data-active={activeId === null}
      >
        <span>All Documents</span>
        <span className={`mono ${styles.allDocsCount}`}>{totalDocs}</span>
      </button>

      <div className={styles.list}>
        {collections.map((c, idx) => (
          <div
            key={c.id}
            className={styles.item}
          >
            {renamingId === c.id ? (
              <input
                ref={renameInputRef}
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={() => submitRename(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submitRename(c.id);
                  if (e.key === "Escape") setRenamingId(null);
                }}
                className={styles.renameInput}
              />
            ) : (
              <button
                onClick={() => onSelect(c.id)}
                onDoubleClick={() => { setRenamingId(c.id); setRenameValue(c.name); }}
                className={styles.itemBtn}
                data-active={activeId === c.id}
              >
                <div className={styles.itemLeft}>
                  <span className={styles.dot} style={{ background: collectionDots[idx % collectionDots.length] }} />
                  <span className={styles.itemName}>{c.name}</span>
                </div>
                <span className={`mono ${styles.itemCount}`}>{c.doc_count}</span>
              </button>
            )}

            {renamingId !== c.id && (
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
                className={styles.deleteBtn}
              >
                <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        ))}
      </div>

      <div className={styles.createWrap}>
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
            className={styles.createInput}
          />
        ) : (
          <button onClick={() => setCreating(true)} className={styles.newBtn}>
            <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
            New collection
          </button>
        )}
      </div>
    </aside>
  );
}
