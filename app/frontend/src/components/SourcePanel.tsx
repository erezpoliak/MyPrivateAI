import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Source } from "../api";

interface Props {
  sources: Source[];
  activeMarker: number | null;
  itemRefs: Map<number, HTMLElement>;
}

function pageRange(src: Source): string {
  if (src.page_start === null) return "";
  if (src.page_end === null || src.page_end === src.page_start) return `p. ${src.page_start}`;
  return `pp. ${src.page_start}–${src.page_end}`;
}

export default function SourcePanel({ sources, activeMarker, itemRefs }: Props) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  function toggleExpanded(chunkIndex: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(chunkIndex)) next.delete(chunkIndex);
      else next.add(chunkIndex);
      return next;
    });
  }

  return (
    <div className="border-t border-gray-100 px-4 py-3 space-y-2">
      <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">
        Sources
      </p>
      {sources.map((src) => {
        const isActive = activeMarker === src.chunk_index;
        const isExpanded = expanded.has(src.chunk_index);
        const range = pageRange(src);

        return (
          <div
            key={src.chunk_index}
            ref={(el) => {
              if (el) itemRefs.set(src.chunk_index, el);
              else itemRefs.delete(src.chunk_index);
            }}
            onClick={() => toggleExpanded(src.chunk_index)}
            className={[
              "rounded-lg border px-3 py-2 text-xs transition-colors cursor-pointer select-none",
              isActive
                ? "border-indigo-300 bg-indigo-50 hover:bg-indigo-100"
                : "border-gray-100 bg-gray-50 hover:bg-gray-100",
            ].join(" ")}
          >
            <div className="flex items-baseline gap-2 mb-1">
              <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-indigo-100 text-indigo-700 text-[10px] font-semibold flex-shrink-0">
                {src.chunk_index}
              </span>
              <span className="font-medium text-gray-700 truncate">{src.title}</span>
              {range && (
                <span className="flex-shrink-0 text-gray-400">{range}</span>
              )}
              {src.snippet && (
                <span className="ml-auto flex-shrink-0 text-gray-400">
                  {isExpanded ? "▲" : "▼"}
                </span>
              )}
            </div>
            {src.snippet && (
              <div
                className={[
                  "text-gray-500 leading-relaxed pl-6 text-xs",
                  isExpanded ? "" : "line-clamp-2",
                ].join(" ")}
              >
                <ReactMarkdown
                  components={{
                    h1: ({ children }) => <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "#374151", marginTop: "0.5rem" }}>{children}</p>,
                    h2: ({ children }) => <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "#374151", marginTop: "0.5rem" }}>{children}</p>,
                    h3: ({ children }) => <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "#374151", marginTop: "0.5rem" }}>{children}</p>,
                  }}
                >
                  {src.snippet
                    .replace(/-<br\s*\/?>\s*/gi, "-")
                    .replace(/<br\s*\/?>\s*/gi, " ")
                    .replace(/- *\n\n+(\S)/g, "$1")
                    .replace(/\|/g, "\n\n---\n\n")}
                </ReactMarkdown>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
