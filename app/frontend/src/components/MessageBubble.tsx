import type { TracePayload } from "../hooks/useSSE";
import TraceTimeline from "./TraceTimeline";
import styles from "./MessageBubble.module.css";

export interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  traces?: TracePayload[];
  isStreaming?: boolean;
  activeMarker?: number | null;
  onMarkerClick?: (n: number) => void;
}

function critiquePending(traces: TracePayload[]): boolean {
  const synthDoneCount    = traces.filter((t) => t.step === "synthesize" && t.status === "done").length;
  const critiqueDoneCount = traces.filter((t) => t.step === "critique" && (t.status === "done" || t.status === "error")).length;
  return synthDoneCount > critiqueDoneCount;
}

function parseContent(text: string): Array<{ type: "text"; value: string } | { type: "marker"; n: number }> {
  return text.split(/(\[\d+\])/g).map((part) => {
    const m = part.match(/^\[(\d+)\]$/);
    return m ? { type: "marker", n: parseInt(m[1], 10) } : { type: "text", value: part };
  });
}

function AnswerSection({
  content, isStreaming, pending, activeMarker, onMarkerClick,
}: {
  content: string;
  isStreaming: boolean;
  pending: boolean;
  activeMarker: number | null;
  onMarkerClick: (n: number) => void;
}) {
  if (!content) {
    return (
      <div className={styles.answer}>
        {isStreaming && <span className={styles.cursor}>▍</span>}
      </div>
    );
  }

  const parts = parseContent(content);

  return (
    <div className={styles.answer} data-pending={pending}>
      {parts.map((part, i) => {
        if (part.type === "text") return <span key={i}>{part.value}</span>;
        return (
          <sup key={i} style={{ lineHeight: 1 }}>
            <button
              onClick={() => onMarkerClick(part.n)}
              className={styles.marker}
              data-active={activeMarker === part.n}
            >
              {part.n}
            </button>
          </sup>
        );
      })}
      {isStreaming && <span className={styles.cursor}>▍</span>}
    </div>
  );
}

export default function MessageBubble({
  role, content, traces = [], isStreaming = false,
  activeMarker = null, onMarkerClick = () => {},
}: MessageBubbleProps) {
  if (role === "user") {
    return (
      <div className={styles.bubble}>
        <div className={styles.userAvatar}>USR</div>
        <div className={styles.userContent}>{content}</div>
      </div>
    );
  }

  const pending = critiquePending(traces);

  return (
    <div className={styles.bubble}>
      <div className={styles.rail}>
        <div className={styles.aiAvatar} />
        <div className={styles.railLine} />
      </div>

      <div className={styles.body}>
        {traces.length > 0 && (
          <TraceTimeline traces={traces} isStreaming={isStreaming} />
        )}

        <AnswerSection
          content={content}
          isStreaming={isStreaming}
          pending={pending}
          activeMarker={activeMarker}
          onMarkerClick={onMarkerClick}
        />

        {pending && (
          <div className={styles.pendingBanner}>
            <span className={styles.spinner} />
            Evaluating…
          </div>
        )}
      </div>
    </div>
  );
}
