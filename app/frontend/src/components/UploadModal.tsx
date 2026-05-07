import { useState, useRef, useEffect } from "react";
import type { Collection } from "../api";
import styles from "./UploadModal.module.css";

interface Props {
  file: File | null;
  collections: Collection[];
  onClose: () => void;
  onIngest: (file: File, collectionId: string | null) => void;
  onCreateCollection: (name: string) => Promise<string>;
  isPending?: boolean;
  collectionDots: string[];
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function RadioCircle({ selected }: { selected: boolean }) {
  return <div className={styles.radioCircle} data-selected={selected} />;
}

export default function UploadModal({ file: initialFile, collections, onClose, onIngest, onCreateCollection, isPending, collectionDots }: Props) {
  const [localFile, setLocalFile] = useState<File | null>(initialFile);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [creatingCollection, setCreatingCollection] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [creatingPending, setCreatingPending] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const newColInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (creatingCollection) newColInputRef.current?.focus();
  }, [creatingCollection]);

  function handleIngest() {
    if (!localFile) return;
    onIngest(localFile, selectedId);
  }

  async function handleCreateCollection() {
    const name = newCollectionName.trim();
    if (!name) return;
    setCreatingPending(true);
    try {
      const id = await onCreateCollection(name);
      setSelectedId(id);
      setCreatingCollection(false);
      setNewCollectionName("");
    } finally {
      setCreatingPending(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={() => { if (!isPending) onClose(); }}>
      <div className={`modal-enter ${styles.modal}`} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.headerTitle}>Upload document</div>
          <button onClick={onClose} className={styles.closeBtn}>×</button>
        </div>

        <div className={styles.fileRow}>
          {localFile ? (
            <div className={styles.fileInfo}>
              <div className={styles.thumb}>
                <div className={styles.thumbLine1} />
                <div className={styles.thumbLine2} />
                <div className={styles.thumbLine3} />
                <svg className={styles.thumbIcon} width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="var(--fg-3)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6z" />
                  <path d="M14 3v6h6" />
                </svg>
              </div>
              <div className={styles.fileMeta}>
                <div className={styles.fileName}>{localFile.name}</div>
                <div className={`mono ${styles.fileSize}`}>{formatSize(localFile.size)} · ready to ingest</div>
              </div>
              <button onClick={() => setLocalFile(null)} className={styles.clearBtn}>×</button>
            </div>
          ) : (
            <div>
              <input ref={fileInputRef} type="file" accept=".pdf" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) setLocalFile(f); }} />
              <button onClick={() => fileInputRef.current?.click()} className={styles.selectBtn}>
                <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 16V4M6 10l6-6 6 6" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
                </svg>
                Click to select a PDF
              </button>
            </div>
          )}
        </div>

        <div className={styles.collectionSection}>
          <div className={`mono ${styles.collectionLabel}`}>Add to collection</div>

          <div className={styles.collectionList}>
            <label className={styles.radioItem} data-selected={selectedId === null}>
              <RadioCircle selected={selectedId === null} />
              <input type="radio" name="collection" style={{ display: "none" }} checked={selectedId === null} onChange={() => setSelectedId(null)} />
              <span className={styles.radioItemName}>All Documents</span>
              <span className={`mono ${styles.radioItemCount}`}>{collections.reduce((s, c) => s + c.doc_count, 0)}</span>
            </label>

            {collections.map((c, idx) => (
              <label key={c.id} className={styles.radioItem} data-selected={selectedId === c.id}>
                <RadioCircle selected={selectedId === c.id} />
                <input type="radio" name="collection" style={{ display: "none" }} checked={selectedId === c.id} onChange={() => setSelectedId(c.id)} />
                <span className={styles.collectionDot} style={{ background: collectionDots[idx % collectionDots.length] }} />
                <span className={styles.radioItemName}>{c.name}</span>
                <span className={`mono ${styles.radioItemCount}`}>{c.doc_count}</span>
              </label>
            ))}

            {creatingCollection ? (
              <div className={styles.newColRow}>
                <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="var(--fg-3)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
                </svg>
                <input
                  ref={newColInputRef}
                  value={newCollectionName}
                  onChange={(e) => setNewCollectionName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleCreateCollection();
                    if (e.key === "Escape") { setCreatingCollection(false); setNewCollectionName(""); }
                  }}
                  placeholder="Collection name…"
                  className={styles.newColInput}
                />
                <button
                  onClick={handleCreateCollection}
                  disabled={creatingPending || !newCollectionName.trim()}
                  className={styles.newColCreate}
                >
                  Create
                </button>
              </div>
            ) : (
              <button onClick={() => setCreatingCollection(true)} className={styles.newColBtn}>
                <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 5v14M5 12h14" />
                </svg>
                New collection
              </button>
            )}
          </div>
        </div>

        <div className={styles.footer}>
          <span className={`mono ${styles.footerNote}`}>🔒 processed locally · never leaves your device</span>
          <button onClick={onClose} disabled={isPending} className={styles.cancelBtn}>Cancel</button>
          <button onClick={handleIngest} disabled={!localFile || isPending} className={styles.ingestBtn}>
            <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 16V4M6 10l6-6 6 6" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
            {isPending ? "Ingesting…" : "Ingest"}
          </button>
        </div>
      </div>
    </div>
  );
}
