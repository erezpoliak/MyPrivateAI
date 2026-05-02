import { useCallback, useRef } from "react";
import type { Source } from "../api";

// ---------------------------------------------------------------------------
// Event payload types
// ---------------------------------------------------------------------------

export interface TracePayload {
  step: string;
  status: string;
  info: string;
}

export interface TokenPayload {
  text: string;
}

export interface DonePayload {
  message_id: string;
  sources: Omit<Source, "id" | "message_id">[];
}

export interface SSECallbacks {
  onTrace: (payload: TracePayload) => void;
  onToken: (payload: TokenPayload) => void;
  onDone: (payload: DonePayload) => void;
  onError?: (err: Error) => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSSE() {
  const abortRef = useRef<AbortController | null>(null);

  const stream = useCallback(async (response: Response, callbacks: SSECallbacks) => {
    if (!response.body) {
      callbacks.onError?.(new Error("Response has no body"));
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE blocks are separated by double newlines.
        const blocks = buffer.split("\n\n");
        // The last element may be an incomplete block — keep it in the buffer.
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          const line = block.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;

          const json = line.slice("data:".length).trim();
          let event: { type: string } & Record<string, unknown>;
          try {
            event = JSON.parse(json);
          } catch {
            continue;
          }

          if (event.type === "trace") {
            callbacks.onTrace(event as unknown as TracePayload);
          } else if (event.type === "token") {
            callbacks.onToken(event as unknown as TokenPayload);
          } else if (event.type === "done") {
            callbacks.onDone(event as unknown as DonePayload);
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        callbacks.onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    } finally {
      reader.releaseLock();
      abortRef.current = null;
    }
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  return { stream, abort };
}
