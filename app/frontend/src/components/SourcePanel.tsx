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
  return (
    <div className="border-t border-gray-100 px-4 py-3 space-y-2">
      <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">
        Sources
      </p>
      {sources.map((src) => {
        const isActive = activeMarker === src.chunk_index;
        const range = pageRange(src);

        return (
          <div
            key={src.chunk_index}
            ref={(el) => {
              if (el) itemRefs.set(src.chunk_index, el);
              else itemRefs.delete(src.chunk_index);
            }}
            className={[
              "rounded-lg border px-3 py-2 text-xs transition-colors",
              isActive
                ? "border-indigo-300 bg-indigo-50"
                : "border-gray-100 bg-gray-50",
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
            </div>
            {src.snippet && (
              <p className="text-gray-500 line-clamp-2 leading-relaxed pl-6">
                {src.snippet}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
