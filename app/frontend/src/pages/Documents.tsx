import { useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { uploadDocument, listDocuments, deleteDocument, type Document } from "../api";
import UploadZone from "../components/UploadZone";
import DocumentsTable from "../components/DocumentsTable";

export default function Documents() {
  const queryClient = useQueryClient();

  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    refetchInterval: (query) =>
      query.state.data?.some((d) => d.status === "ingesting") ? 3000 : false,
  });

  const { mutate: upload, isPending } = useMutation({
    mutationFn: uploadDocument,
    onSuccess: (doc) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`"${doc.title}" uploaded — ingesting…`);
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

  return (
    <div className="p-8 max-w-4xl mx-auto w-full overflow-y-auto h-full">
      <h1 className="text-2xl font-semibold text-gray-800 mb-6">Documents</h1>
      <UploadZone onFiles={handleFiles} isPending={isPending} />
      <DocumentsTable docs={docs} onDelete={remove} />
    </div>
  );
}
