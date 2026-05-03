import { useRef, useState, useCallback, useId } from "react";

export default function UploadZone({
  onFiles,
  isPending,
}: {
  onFiles: (files: File[]) => void;
  isPending: boolean;
}) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list) return;
      onFiles(Array.from(list));
    },
    [onFiles]
  );

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current += 1;
    if (dragCounter.current === 1) setDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) setDragging(false);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  return (
    <>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={[
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed",
          "cursor-pointer select-none transition-colors py-16 px-8 text-center",
          dragging
            ? "border-indigo-500 bg-indigo-50"
            : "border-gray-300 bg-gray-50 hover:border-indigo-400 hover:bg-indigo-50",
          isPending ? "pointer-events-none opacity-60" : "",
        ].join(" ")}
      >
        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0-3 3m3-3 3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.338-2.32 5.75 5.75 0 0 1 1.023 11.095" />
        </svg>
        <p className="text-sm text-gray-600">
          <span className="font-medium text-indigo-600">Click to upload</span>
          {" or drag and drop"}
        </p>
        <p className="text-xs text-gray-400">PDF files only</p>
        {isPending && (
          <p className="text-xs text-indigo-500 animate-pulse">Uploading…</p>
        )}
      </div>
    </>
  );
}
