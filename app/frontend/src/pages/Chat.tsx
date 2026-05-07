import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { listChats, createChat, getChat, postMessage, deleteChat, type Message, type Source } from "../api";
import { useSSE, type DonePayload, type TracePayload } from "../hooks/useSSE";
import ChatSidebar from "../components/ChatSidebar";
import ChatInput from "../components/ChatInput";
import MessageBubble from "../components/MessageBubble";
import SourcePanel from "../components/SourcePanel";
import styles from "./Chat.module.css";

interface StreamingMessage {
  role: "assistant";
  content: string;
  traces: TracePayload[];
  isStreaming: true;
}

type DisplayMessage = Message | StreamingMessage;

function isStreaming(m: DisplayMessage): m is StreamingMessage {
  return "isStreaming" in m;
}

export default function Chat() {
  const queryClient = useQueryClient();
  const { stream } = useSSE();

  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingMsg, setStreamingMsg] = useState<StreamingMessage | null>(null);
  const [activeMarker, setActiveMarker] = useState<number | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sourceItemRefs = useRef<Map<number, HTMLElement>>(new Map());

  const { data: chats = [], isLoading: chatsLoading } = useQuery({
    queryKey: ["chats"],
    queryFn: listChats,
  });

  const { data: chatDetail } = useQuery({
    queryKey: ["chat", activeChatId],
    queryFn: () => getChat(activeChatId!),
    enabled: activeChatId !== null,
  });

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [chatDetail?.messages, streamingMsg?.content, scrollToBottom]);

  const displaySources = useMemo((): Source[] => {
    const msgs = chatDetail?.messages ?? [];
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && msgs[i].sources?.length > 0) {
        return msgs[i].sources;
      }
    }
    return [];
  }, [chatDetail?.messages]);

  function handleMarkerClick(n: number) {
    setActiveMarker((prev) => (prev === n ? null : n));
    if (!inspectorOpen) setInspectorOpen(true);
    const el = sourceItemRefs.current.get(n);
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function handleDeleteChat(chatId: string) {
    try {
      await deleteChat(chatId);
      await queryClient.invalidateQueries({ queryKey: ["chats"] });
      if (activeChatId === chatId) setActiveChatId(null);
      toast.success("Thread deleted");
    } catch (err) {
      toast.error((err as Error).message ?? "Failed to delete thread");
    }
  }

  async function handleNewChat() {
    try {
      const chat = await createChat();
      await queryClient.invalidateQueries({ queryKey: ["chats"] });
      setActiveChatId(chat.id);
    } catch (err) {
      toast.error((err as Error).message ?? "Failed to create thread");
    }
  }

  async function handleSend() {
    if (!input.trim() || sending) return;

    let chatId = activeChatId;
    if (!chatId) {
      try {
        const chat = await createChat();
        await queryClient.invalidateQueries({ queryKey: ["chats"] });
        setActiveChatId(chat.id);
        chatId = chat.id;
      } catch (err) {
        toast.error((err as Error).message ?? "Failed to create thread");
        return;
      }
    }

    const userContent = input.trim();
    setInput("");
    setSending(true);
    setStreamingMsg({ role: "assistant", content: "", traces: [], isStreaming: true });

    queryClient.setQueryData(["chat", chatId], (old: typeof chatDetail) => {
      if (!old) return old;
      const tempUser: Message = {
        id: `temp-${Date.now()}`,
        chat_id: chatId!,
        role: "user",
        content: userContent,
        created_at: new Date().toISOString(),
        sources: [],
        traces: [],
      };
      return { ...old, messages: [...old.messages, tempUser] };
    });

    try {
      const response = await postMessage(chatId, userContent);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      let accumulated = "";

      await stream(response, {
        onToken: ({ text }) => {
          accumulated += text;
          setStreamingMsg((prev) => (prev ? { ...prev, content: accumulated } : prev));
        },
        onTrace: (payload) => {
          setStreamingMsg((prev) =>
            prev ? { ...prev, traces: [...prev.traces, payload] } : prev
          );
        },
        onReset: () => {
          accumulated = "";
          setStreamingMsg((prev) => (prev ? { ...prev, content: "" } : prev));
        },
        onDone: (payload: DonePayload) => {
          const finalMessage: Message = {
            id: payload.message_id,
            chat_id: chatId!,
            role: "assistant",
            content: accumulated,
            created_at: new Date().toISOString(),
            sources: payload.sources.map((s, i) => ({ ...s, id: i, message_id: payload.message_id })),
            traces: payload.traces ?? [],
          };
          queryClient.setQueryData(["chat", chatId], (old: typeof chatDetail) => {
            if (!old) return old;
            return { ...old, messages: [...old.messages, finalMessage] };
          });
          setStreamingMsg(null);
          queryClient.invalidateQueries({ queryKey: ["chat", chatId] });
          queryClient.invalidateQueries({ queryKey: ["chats"] });
        },
        onError: (err) => toast.error(err.message ?? "Streaming error"),
      });
    } catch (err) {
      toast.error((err as Error).message ?? "Failed to send message");
    } finally {
      setSending(false);
      setStreamingMsg(null);
    }
  }

  const messages: DisplayMessage[] = [
    ...(chatDetail?.messages ?? []),
    ...(streamingMsg ? [streamingMsg] : []),
  ];

  const activeChat = chatDetail ?? chats.find((c) => c.id === activeChatId);
  const messageCount = chatDetail?.messages?.length ?? 0;

  return (
    <div className={styles.root}>
      <ChatSidebar
        chats={chats}
        loading={chatsLoading}
        activeChatId={activeChatId}
        onSelect={setActiveChatId}
        onNewChat={handleNewChat}
        onDelete={handleDeleteChat}
      />

      <main className={styles.main}>
        <div className={styles.threadHeader}>
          <div className={`mono ${styles.threadEyebrow}`}>
            Thread · {messageCount} message{messageCount !== 1 ? "s" : ""}
          </div>
          <div className={styles.threadTitle}>
            <span className={styles.threadTitleText}>
              {activeChat?.title || (activeChatId ? "New thread" : "Select or start a thread")}
            </span>
          </div>
        </div>

        <div className={styles.messages}>
          {messages.length === 0 && (
            <div className={styles.emptyState}>
              <p className={`mono ${styles.emptyText}`}>
                {activeChatId ? "No messages yet — ask something below." : "Select a thread or start a new one."}
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble
              key={isStreaming(msg) ? "streaming" : (msg as Message).id ?? i}
              role={msg.role}
              content={msg.content}
              traces={isStreaming(msg) ? msg.traces : (msg as Message).traces ?? []}
              isStreaming={isStreaming(msg)}
              activeMarker={activeMarker}
              onMarkerClick={handleMarkerClick}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput value={input} disabled={sending} onChange={setInput} onSend={handleSend} />
      </main>

      <SourcePanel
        sources={displaySources}
        activeMarker={activeMarker}
        itemRefs={sourceItemRefs.current}
        open={inspectorOpen}
        onToggle={() => setInspectorOpen((o) => !o)}
      />

      {!inspectorOpen && (
        <button onClick={() => setInspectorOpen(true)} className={styles.inspectorTab}>
          <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
            <path d="m15 6-6 6 6 6" />
          </svg>
        </button>
      )}
    </div>
  );
}
