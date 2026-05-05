import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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
import UploadZone from "../components/UploadZone";
import DocumentsTable from "../components/DocumentsTable";
import CollectionSidebar from "../components/CollectionSidebar";
import CollectionPickerModal from "../components/CollectionPickerModal";

export default function Documents() {
  const queryClient = useQueryClient();
  const [activeCollection, setActiveCollection] = useState<string | null>(null);
  const [uploadedDoc, setUploadedDoc] = useState<Document | null>(null);

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

  const { mutate: upload, isPending } = useMutation({
    mutationFn: uploadDocument,
    onSuccess: (doc) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`"${doc.title}" uploaded — ingesting…`);
      if (collections.length > 0) setUploadedDoc(doc);
    },
    onError: (err: Error) => {
      toast.error(err.message || "Upload failed");
    },
  });

  const { mutate: remove } = useMutation({
    mutationFn: deleteDocument,
    onMutate: async (docId) => {
      await queryClient.cancelQueries({ queryKey: ["documents"] });
      const prev = queryClient.getQueryData<Document[]>(["documents"]);
      queryClient.setQueryData<Document[]>(["documents"], (old) =>
        old?.filter((d) => d.id !== docId) ?? []
      );
      return { prev };
    },
    onSuccess: (_data, docId, ctx) => {
      const doc = ctx?.prev?.find((d) => d.id === docId);
      toast.success(`"${doc?.title ?? "Document"}" deleted`);
    },
    onError: (err: Error, _docId, ctx) => {
      queryClient.setQueryData(["documents"], ctx?.prev);
      toast.error(err.message || "Delete failed");
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  const { mutate: addCollection } = useMutation({
    mutationFn: (name: string) => createCollection(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["collections"] }),
    onError: (err: Error) => toast.error(err.message || "Failed to create collection"),
  });

  const { mutate: updateCollection } = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameCollection(id, name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["collections"] }),
    onError: (err: Error) => toast.error(err.message || "Failed to rename collection"),
  });

  const { mutate: removeCollection } = useMutation({
    mutationFn: deleteCollection,
    onMutate: (id) => {
      if (activeCollection === id) setActiveCollection(null);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (err: Error) => toast.error(err.message || "Failed to delete collection"),
  });

  const { mutate: assign } = useMutation({
    mutationFn: ({ docId, collectionId }: { docId: string; collectionId: string }) =>
      addDocToCollection(collectionId, docId),
    onMutate: async ({ docId, collectionId }) => {
      await queryClient.cancelQueries({ queryKey: ["documents"] });
      const prev = queryClient.getQueryData<Document[]>(["documents"]);
      queryClient.setQueryData<Document[]>(["documents"], (old) =>
        old?.map((d) =>
          d.id === docId
            ? { ...d, collection_ids: [...new Set([...d.collection_ids, collectionId])] }
            : d
        ) ?? []
      );
      return { prev };
    },
    onError: (err: Error, _vars, ctx) => {
      queryClient.setQueryData(["documents"], ctx?.prev);
      toast.error(err.message || "Failed to assign");
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  const { mutate: unassign } = useMutation({
    mutationFn: ({ docId, collectionId }: { docId: string; collectionId: string }) =>
      removeDocFromCollection(collectionId, docId),
    onMutate: async ({ docId, collectionId }) => {
      await queryClient.cancelQueries({ queryKey: ["documents"] });
      const prev = queryClient.getQueryData<Document[]>(["documents"]);
      queryClient.setQueryData<Document[]>(["documents"], (old) =>
        old?.map((d) =>
          d.id === docId
            ? { ...d, collection_ids: d.collection_ids.filter((id) => id !== collectionId) }
            : d
        ) ?? []
      );
      return { prev };
    },
    onError: (err: Error, _vars, ctx) => {
      queryClient.setQueryData(["documents"], ctx?.prev);
      toast.error(err.message || "Failed to unassign");
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  const handleFiles = useCallback(
    (files: File[]) => {
      const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
      if (pdfs.length === 0) {
        toast.error("Only PDF files are accepted");
        return;
      }
      pdfs.forEach((f) => upload(f));
    },
    [upload]
  );

  const visibleDocs =
    activeCollection === null
      ? docs
      : docs.filter((d) => d.collection_ids.includes(activeCollection));

  return (
    <div className="flex h-full overflow-hidden">
      {uploadedDoc && (
        <CollectionPickerModal
          doc={uploadedDoc}
          collections={collections}
          onConfirm={(collectionIds) => {
            collectionIds.forEach((cid) => assign({ docId: uploadedDoc.id, collectionId: cid }));
            setUploadedDoc(null);
          }}
          onSkip={() => setUploadedDoc(null)}
        />
      )}
      <CollectionSidebar
        collections={collections}
        activeId={activeCollection}
        onSelect={setActiveCollection}
        onCreate={(name) => addCollection(name)}
        onRename={(id, name) => updateCollection({ id, name })}
        onDelete={(id) => removeCollection(id)}
        onDocDrop={(docId, collectionId) => assign({ docId, collectionId })}
        totalDocs={docs.length}
      />
      <div className="flex-1 p-8 overflow-y-auto">
        <h1 className="text-2xl font-semibold text-gray-800 mb-6">
          {activeCollection
            ? (collections.find((c) => c.id === activeCollection)?.name ?? "Collection")
            : "Documents"}
        </h1>
        <UploadZone onFiles={handleFiles} isPending={isPending} />
        <DocumentsTable
          docs={visibleDocs}
          collections={collections}
          onDelete={remove}
          onAssign={(docId, collectionId) => assign({ docId, collectionId })}
          onUnassign={(docId, collectionId) => unassign({ docId, collectionId })}
        />
      </div>
    </div>
  );
}
