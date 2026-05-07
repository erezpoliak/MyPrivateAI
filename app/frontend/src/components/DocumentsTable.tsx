import { useState, useRef, useEffect } from "react";
import type { Collection, Document } from "../api";
import StatusBadge from "./StatusBadge";
import styles from "./DocumentsTable.module.css";

function MoreMenu({
  doc, collections, collectionDots, onDelete, onAssign, onUnassign, onClose,
}: {
  doc: Document; collections: Collection[];
  collectionDots: string[];
  onDelete: () => void;
  onAssign: (cid: string) => void;
  onUnassign: (cid: string) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<"main" | "collections">("main");

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div ref={ref} className={styles.menu}>
      {view === "main" ? (
        <>
          <button className={styles.menuItem} onClick={() => setView("collections")}>
            <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
            </svg>
            Move to collection
          </button>
          <div className={styles.menuDivider} />
          <button className={`${styles.menuItem} ${styles.menuDelete}`} onClick={onDelete}>
            <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13" />
            </svg>
            Delete
          </button>
        </>
      ) : (
        <>
          <button className={`${styles.menuItem} ${styles.menuBack}`} onClick={() => setView("main")}>
            <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="m15 6-6 6 6 6" />
            </svg>
            Back
          </button>
          <div className={styles.menuDivider} />
          {collections.length === 0 ? (
            <p className={styles.menuEmpty}>No collections yet.</p>
          ) : (
            collections.map((c, idx) => {
              const assigned = doc.collection_ids.includes(c.id);
              return (
                <button
                  key={c.id}
                  className={`${styles.menuItem} ${styles.menuCollectionItem}`}
                  onClick={() => { assigned ? onUnassign(c.id) : onAssign(c.id); onClose(); }}
                >
                  <div className={styles.menuCollectionLeft}>
                    <span className={styles.menuDot} style={{ background: collectionDots[idx % collectionDots.length] }} />
                    <span>{c.name}</span>
                  </div>
                  {assigned && (
                    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                      <path d="m4 12 5 5L20 6" />
                    </svg>
                  )}
                </button>
              );
            })
          )}
        </>
      )}
    </div>
  );
}

function DocCard({
  doc, collections, collectionMap, collectionDots, onDelete, onAssign, onUnassign,
}: {
  doc: Document;
  collections: Collection[];
  collectionMap: Record<string, { name: string; idx: number }>;
  collectionDots: string[];
  onDelete: () => void;
  onAssign: (cid: string) => void;
  onUnassign: (cid: string) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className={styles.card}>
      <div className={styles.thumb}>
        <div className={styles.thumbLine1} />
        <div className={styles.thumbLine2} />
        <div className={styles.thumbLine3} />
        <svg className={styles.thumbIcon} width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="var(--fg-3)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6z" />
          <path d="M14 3v6h6M9 13h6M9 17h4" />
        </svg>
      </div>

      <div className={styles.info}>
        <div className={styles.docTitle}>{doc.title}</div>
        <div className={`mono ${styles.meta}`}>
          <span>{doc.page_count || "—"} pp</span>
          <span>{doc.chunk_count || "—"} chunks</span>
          <span>{new Date(doc.created_at).toLocaleDateString("en-US", { month: "numeric", day: "numeric", year: "2-digit" })}</span>
        </div>
        <div className={styles.badges}>
          <StatusBadge status={doc.status} />
          {doc.collection_ids.map((cid) => {
            const c = collectionMap[cid];
            if (!c) return null;
            return (
              <span key={cid} className={`mono ${styles.collectionChip}`}>{c.name}</span>
            );
          })}
        </div>
      </div>

      <div className={styles.menuWrap}>
        <button className={styles.dotsBtn} onClick={() => setMenuOpen((o) => !o)}>
          <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
            <circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" />
            <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
            <circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />
          </svg>
        </button>
        {menuOpen && (
          <MoreMenu
            doc={doc} collections={collections} collectionDots={collectionDots}
            onDelete={() => { onDelete(); setMenuOpen(false); }}
            onAssign={onAssign} onUnassign={onUnassign}
            onClose={() => setMenuOpen(false)}
          />
        )}
      </div>
    </div>
  );
}

export default function DocumentsTable({
  docs, collections, collectionDots, onDelete, onAssign, onUnassign,
}: {
  docs: Document[];
  collections: Collection[];
  collectionDots: string[];
  onDelete: (id: string) => void;
  onAssign: (docId: string, collectionId: string) => void;
  onUnassign: (docId: string, collectionId: string) => void;
}) {
  const collectionMap = Object.fromEntries(
    collections.map((c, idx) => [c.id, { name: c.name, idx }])
  );

  if (docs.length === 0) {
    return <p className={`mono ${styles.empty}`}>No documents yet — upload a PDF above.</p>;
  }

  return (
    <div className={styles.grid}>
      {docs.map((doc) => (
        <DocCard
          key={doc.id}
          doc={doc}
          collections={collections}
          collectionMap={collectionMap}
          collectionDots={collectionDots}
          onDelete={() => onDelete(doc.id)}
          onAssign={(cid) => onAssign(doc.id, cid)}
          onUnassign={(cid) => onUnassign(doc.id, cid)}
        />
      ))}
    </div>
  );
}
