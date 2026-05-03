import { useState, useRef } from "react";
import type { Source } from "../api";
import type { TracePayload } from "../hooks/useSSE";
import TraceTimeline from "./TraceTimeline";
import SourcePanel from "./SourcePanel";

export interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  traces?: TracePayload[];
  isStreaming?: boolean;
}

export default function MessageBubble({
  role,
  content,
  sources = [],
  traces = [],
  isStreaming = false,
}: MessageBubbleProps) {
  const [activeMarker, setActiveMarker] = useState<number | null>(null);
  // Refs keyed by 1-based citation index — populated by SourcesSection, read here
  const sourceItemRefs = useRef<Map<number, HTMLElement>>(new Map());

  if (role === "user") {
    return (
      <div className="flex justify-end mb-3">
        <div className="max-w-[70%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap bg-indigo-600 text-white">
          {content}
        </div>
      </div>
    );
  }

  function handleMarkerClick(n: number) {
    setActiveMarker((prev) => (prev === n ? null : n));
    const el = sourceItemRefs.current.get(n);
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[75%] rounded-2xl border border-gray-200 bg-white text-sm text-gray-800 overflow-hidden">
        <TraceTimeline traces={traces} isStreaming={isStreaming} />

        <AnswerSection
          content={content}
          isStreaming={isStreaming}
          activeMarker={activeMarker}
          onMarkerClick={handleMarkerClick}
        />

        {sources.length > 0 && (
          <SourcePanel
            sources={sources}
            activeMarker={activeMarker}
            itemRefs={sourceItemRefs.current}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Answer section
// ---------------------------------------------------------------------------

function parseContent(text: string): Array<{ type: "text"; value: string } | { type: "marker"; n: number }> {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) return { type: "marker", n: parseInt(match[1], 10) };
    return { type: "text", value: part };
  });
}

function AnswerSection({
  content,
  isStreaming,
  activeMarker,
  onMarkerClick,
}: {
  content: string;
  isStreaming: boolean;
  activeMarker: number | null;
  onMarkerClick: (n: number) => void;
}) {
  if (!content) {
    return (
      <div className="px-4 py-3 leading-relaxed">
        {isStreaming ? <span className="animate-pulse text-gray-400">▍</span> : null}
      </div>
    );
  }

  const parts = parseContent(content);

  return (
    <div className="px-4 py-3 leading-relaxed whitespace-pre-wrap">
      {parts.map((part, i) => {
        if (part.type === "text") return <span key={i}>{part.value}</span>;
        const isActive = activeMarker === part.n;
        return (
          <sup key={i}>
            <button
              onClick={() => onMarkerClick(part.n)}
              className={[
                "inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-semibold leading-none transition-colors mx-0.5",
                isActive
                  ? "bg-indigo-600 text-white"
                  : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200",
              ].join(" ")}
            >
              {part.n}
            </button>
          </sup>
        );
      })}
      {isStreaming && <span className="animate-pulse text-gray-400">▍</span>}
    </div>
  );
}

