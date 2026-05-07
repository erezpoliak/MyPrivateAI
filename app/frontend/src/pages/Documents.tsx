import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  uploadDocument,
  listDocuments,
  deleteDocument,
  listCollections,
  createCollection,
  renameCollection,
  deleteCollection,
  addDocToCollection,
  removeDocFromCollection,
  type Document,
} from "../api";
import DocumentsTable from "../components/DocumentsTable";
import UploadModal from "../components/UploadModal";
import CollectionSidebar from "../components/CollectionSidebar";
import styles from "./Documents.module.css";

const COLLECTION_DOTS = [
  "oklch(0.82 0.13 200)",
  "oklch(0.72 0.16 20)",
  "oklch(0.82 0.14 75)",
  "oklch(0.78 0.14 155)",
  "var(--accent)",
];

export default function Documents() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    refetchInterval: (query) =>
      query.state.data?.some((d) => d.status === "ingesting") ? 3000 : false,
  });

  const { data: collections = [] } = useQuery({
    queryKey: ["collections"],
    queryFn: listCollections,
  });

  async function handleIngest(file: File, collectionId: string | null) {
    setIsUploading(true);
    try {
      const doc = await uploadDocument(file);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`"${doc.title}" uploading…`);
      if (collectionId) {
        await addDocToCollection(collectionId, doc.id);
        queryClient.invalidateQueries({ queryKey: ["documents"] });
        queryClient.invalidateQueries({ queryKey: ["collections"] });
      }
      setShowUploadModal(false);
      setPendingFile(null);
    } catch (err) {
      toast.error((err as Error).message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleCreateCollection(name: string): Promise<string> {
    const coll = await createCollection(name);
    queryClient.invalidateQueries({ queryKey: ["collections"] });
    return coll.id;
  }

  async function handleRenameCollection(id: string, name: string) {
    try {
      await renameCollection(id, name);
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    } catch (err) {
      toast.error((err as Error).message || "Rename failed");
    }
  }

  async function handleDeleteCollection(id: string) {
    try {
      await deleteCollection(id);
      if (activeFilter === id) setActiveFilter(null);
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    } catch (err) {
      toast.error((err as Error).message || "Delete failed");
    }
  }

  function openUploadModal(file?: File) {
    setPendingFile(file ?? null);
    setShowUploadModal(true);
  }

  function handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) openUploadModal(f);
    e.target.value = "";
  }

  function handlePageDrop(e: React.DragEvent) {
    e.preventDefault();
    const pdfs = Array.from(e.dataTransfer.files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) { toast.error("Only PDF files are accepted"); return; }
    openUploadModal(pdfs[0]);
  }

  async function handleDelete(id: string) {
    const doc = docs.find((d) => d.id === id);
    try {
      await deleteDocument(id);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`"${doc?.title ?? "Document"}" deleted`);
    } catch (err) {
      toast.error((err as Error).message || "Delete failed");
    }
  }

  async function handleAssign(docId: string, collectionId: string) {
    await addDocToCollection(collectionId, docId);
    queryClient.invalidateQueries({ queryKey: ["documents"] });
    queryClient.invalidateQueries({ queryKey: ["collections"] });
  }

  async function handleUnassign(docId: string, collectionId: string) {
    await removeDocFromCollection(collectionId, docId);
    queryClient.invalidateQueries({ queryKey: ["documents"] });
    queryClient.invalidateQueries({ queryKey: ["collections"] });
  }

  const visibleDocs: Document[] = activeFilter === null
    ? docs
    : docs.filter((d) => d.collection_ids.includes(activeFilter));

  return (
    <div
      className={styles.root}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handlePageDrop}
    >
      <input ref={fileInputRef} type="file" accept=".pdf" style={{ display: "none" }} onChange={handleFileInputChange} />

      {showUploadModal && (
        <UploadModal
          file={pendingFile}
          collections={collections}
          onClose={() => { setShowUploadModal(false); setPendingFile(null); }}
          onIngest={handleIngest}
          onCreateCollection={handleCreateCollection}
          isPending={isUploading}
          collectionDots={COLLECTION_DOTS}
        />
      )}

      <CollectionSidebar
        collections={collections}
        activeId={activeFilter}
        onSelect={setActiveFilter}
        onCreate={(name) => handleCreateCollection(name).then((id) => setActiveFilter(id))}
        onRename={handleRenameCollection}
        onDelete={handleDeleteCollection}
        totalDocs={docs.length}
        collectionDots={COLLECTION_DOTS}
      />

      <div className={styles.content}>
        <div className={styles.header}>
          <div>
            <div className={`mono ${styles.eyebrow}`}>Library</div>
            <h1 className={styles.h1}>
              Documents{" "}
              <span className={`serif ${styles.h1Count}`}>
                — {docs.length} item{docs.length !== 1 ? "s" : ""}
              </span>
            </h1>
          </div>
          <button onClick={() => fileInputRef.current?.click()} className={styles.uploadBtn}>
            <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 16V4M6 10l6-6 6 6" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
            Upload PDF
          </button>
        </div>

        <div className={styles.dropStrip} onClick={() => openUploadModal()}>
          <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="var(--fg-3)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
            <path d="M7 18a5 5 0 1 1 1-9.9A6 6 0 0 1 19 10a4 4 0 0 1 0 8H7z" />
            <path d="M12 14V8M9 11l3-3 3 3" />
          </svg>
          <span>Drop PDFs anywhere on this page</span>
        </div>

        <DocumentsTable
          docs={visibleDocs}
          collections={collections}
          onDelete={handleDelete}
          onAssign={handleAssign}
          onUnassign={handleUnassign}
          collectionDots={COLLECTION_DOTS}
        />
      </div>
    </div>
  );
}
