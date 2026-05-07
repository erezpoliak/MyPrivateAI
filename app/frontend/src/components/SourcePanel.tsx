import { useState, useEffect } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import type { Source } from "../api";
import styles from "./SourcePanel.module.css";

interface Props {
  sources: Source[];
  activeMarker: number | null;
  itemRefs: Map<number, HTMLElement>;
  open: boolean;
  onToggle: () => void;
}

function pageLabel(src: Source): string {
  if (src.page_start === null) return "";
  if (src.page_end === null || src.page_end === src.page_start) return `p. ${src.page_start}`;
  return `pp. ${src.page_start}–${src.page_end}`;
}

function cleanSnippet(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, "")
    .replace(/-<br\s*\/?>\s*/gi, "-")
    .replace(/<br\s*\/?>\s*/gi, " ")
    .replace(/- *\n\n+(\S)/g, "$1")
    .replace(/\|/g, "\n\n---\n\n");
}

const mdComponents = {
  h1: ({ children }: { children: ReactNode }) => <p style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--fg)", marginTop: "0.75rem", marginBottom: "0.25rem" }}>{children}</p>,
  h2: ({ children }: { children: ReactNode }) => <p style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--fg)", marginTop: "0.75rem", marginBottom: "0.25rem" }}>{children}</p>,
  h3: ({ children }: { children: ReactNode }) => <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--fg-2)", marginTop: "0.5rem", marginBottom: "0.25rem" }}>{children}</p>,
  p:  ({ children }: { children: ReactNode }) => <p style={{ margin: "0.3rem 0", lineHeight: 1.65 }}>{children}</p>,
  li: ({ children }: { children: ReactNode }) => <li style={{ margin: "0.15rem 0" }}>{children}</li>,
  code: ({ children }: { children: ReactNode }) => <code style={{ background: "var(--bg-4)", borderRadius: 3, padding: "1px 4px", fontSize: "0.85em", fontFamily: "var(--mono)" }}>{children}</code>,
};

function ViewerModal({ src, onClose }: { src: Source; onClose: () => void }) {
  const page = pageLabel(src);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={`modal-enter ${styles.modal}`} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <div className={styles.modalBadge}>{src.chunk_index}</div>
          <div className={styles.modalMeta}>
            <div className={styles.modalTitle}>{src.title}</div>
            {page && <div className={`mono ${styles.modalPage}`}>{page}</div>}
          </div>
          <button onClick={onClose} className={styles.modalClose}>×</button>
        </div>
        <div className={styles.modalBody}>
          {src.snippet ? (
            <ReactMarkdown components={mdComponents}>{cleanSnippet(src.snippet)}</ReactMarkdown>
          ) : (
            <p className={`mono ${styles.modalEmpty}`}>No content available.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function SourceCard({
  src, isActive, itemRefs,
}: {
  src: Source;
  isActive: boolean;
  itemRefs: Map<number, HTMLElement>;
}) {
  const [expanded, setExpanded] = useState(false);
  const [viewerOpen, setViewerOpen] = useState(false);
  const page = pageLabel(src);

  return (
    <>
      {viewerOpen && <ViewerModal src={src} onClose={() => setViewerOpen(false)} />}

      <div
        ref={(el) => { if (el) itemRefs.set(src.chunk_index, el); else itemRefs.delete(src.chunk_index); }}
        onClick={() => setExpanded((e) => !e)}
        className={styles.card}
        data-active={isActive}
      >
        <div className={styles.cardTop}>
          <div className={styles.badge} data-active={isActive}>{src.chunk_index}</div>
          {page && <span className={`mono ${styles.page}`}>{page}</span>}
          <span style={{ flex: 1 }} />
          <svg
            width={11} height={11} viewBox="0 0 24 24" fill="none"
            stroke="var(--fg-4)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"
            className={styles.chevron}
            data-expanded={expanded}
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </div>

        <div className={styles.cardTitle} style={{ marginBottom: expanded && src.snippet ? 8 : 0 }}>
          {src.title}
        </div>

        {src.snippet && (
          expanded ? (
            <div className={styles.snippetExpanded}>
              <ReactMarkdown components={mdComponents}>{cleanSnippet(src.snippet)}</ReactMarkdown>
            </div>
          ) : (
            <div className={styles.snippetCollapsed}>
              {src.snippet.replace(/```[\s\S]*?```/g, "").replace(/<br\s*\/?>/gi, " ").replace(/\n+/g, " ")}
            </div>
          )
        )}

        {src.snippet && (
          <button
            onClick={(e) => { e.stopPropagation(); setViewerOpen(true); }}
            className={styles.viewerBtn}
          >
            <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
            </svg>
            Open in viewer
          </button>
        )}
      </div>
    </>
  );
}

export default function SourcePanel({ sources, activeMarker, itemRefs, open, onToggle }: Props) {
  return (
    <aside className={styles.panel} style={{ width: open ? 360 : 0 }}>
      <div className={styles.inner}>
        <div className={styles.header}>
          <span className={`mono ${styles.headerLabel}`}>Sources · {sources.length}</span>
          <button onClick={onToggle} className={styles.collapseBtn}>
            <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="m9 6 6 6-6 6" />
            </svg>
          </button>
        </div>

        <div className={styles.list}>
          {sources.length === 0 && <p className={`mono ${styles.empty}`}>No sources yet</p>}
          {sources.map((src) => (
            <SourceCard
              key={src.chunk_index}
              src={src}
              isActive={activeMarker === src.chunk_index}
              itemRefs={itemRefs}
            />
          ))}
        </div>
      </div>
    </aside>
  );
}
