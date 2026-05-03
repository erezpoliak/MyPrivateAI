import { useState, useRef, useEffect, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { listChats, createChat, getChat, postMessage, deleteChat, type Message } from "../api";
import { useSSE, type DonePayload, type TracePayload } from "../hooks/useSSE";
import ChatSidebar from "../components/ChatSidebar";
import ChatInput from "../components/ChatInput";
import MessageBubble from "../components/MessageBubble";

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

  const messagesEndRef = useRef<HTMLDivElement>(null);

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

  async function handleDeleteChat(chatId: string) {
    try {
      await deleteChat(chatId);
      await queryClient.invalidateQueries({ queryKey: ["chats"] });
      if (activeChatId === chatId) {
        setActiveChatId(null);
      }
      toast.success("Chat deleted");
    } catch (err) {
      toast.error((err as Error).message ?? "Failed to delete chat");
    }
  }

  async function handleNewChat() {
    try {
      const chat = await createChat();
      await queryClient.invalidateQueries({ queryKey: ["chats"] });
      setActiveChatId(chat.id);
    } catch (err) {
      toast.error((err as Error).message ?? "Failed to create chat");
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
        toast.error((err as Error).message ?? "Failed to create chat");
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
        onDone: (payload: DonePayload) => {
          // Write final message to cache immediately so there's no gap when
          // streamingMsg clears before the refetch lands.
          const finalMessage: Message = {
            id: payload.message_id,
            chat_id: chatId!,
            role: "assistant",
            content: accumulated,
            created_at: new Date().toISOString(),
            sources: payload.sources.map((s, i) => ({
              ...s,
              id: i,
              message_id: payload.message_id,
            })),
          };
          queryClient.setQueryData(["chat", chatId], (old: typeof chatDetail) => {
            if (!old) return old;
            return { ...old, messages: [...old.messages, finalMessage] };
          });
          // Sync with server to replace optimistic IDs with canonical ones.
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

  return (
    <div className="flex flex-1 min-h-0">
      <ChatSidebar
        chats={chats}
        loading={chatsLoading}
        activeChatId={activeChatId}
        onSelect={setActiveChatId}
        onNewChat={handleNewChat}
        onDelete={handleDeleteChat}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="h-full flex items-center justify-center text-gray-400 text-sm">
              Select a chat or start a new one
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble
              key={isStreaming(msg) ? "streaming" : (msg as Message).id ?? i}
              role={msg.role}
              content={msg.content}
              sources={isStreaming(msg) ? [] : (msg as Message).sources}
              traces={isStreaming(msg) ? msg.traces : []}
              isStreaming={isStreaming(msg)}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput
          value={input}
          disabled={sending}
          onChange={setInput}
          onSend={handleSend}
        />
      </div>
    </div>
  );
}
