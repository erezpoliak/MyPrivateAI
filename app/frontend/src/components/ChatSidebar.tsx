import type { Chat } from "../api";

function ChatItem({
  chat,
  active,
  onSelect,
}: {
  chat: Chat;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={[
        "w-full text-left px-3 py-2 rounded-md text-sm truncate transition-colors",
        active
          ? "bg-indigo-100 text-indigo-800 font-medium"
          : "text-gray-700 hover:bg-gray-100",
      ].join(" ")}
    >
      {chat.title || "New chat"}
    </button>
  );
}

interface Props {
  chats: Chat[];
  loading: boolean;
  activeChatId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}

export default function ChatSidebar({
  chats,
  loading,
  activeChatId,
  onSelect,
  onNewChat,
}: Props) {
  return (
    <aside className="w-60 flex-shrink-0 border-r border-gray-200 flex flex-col">
      <div className="p-3 border-b border-gray-100">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-sm font-medium rounded-md bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
        >
          <span className="text-base leading-none">+</span> New chat
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {loading && (
          <p className="text-xs text-gray-400 px-2 pt-2">Loading…</p>
        )}
        {chats.map((chat) => (
          <ChatItem
            key={chat.id}
            chat={chat}
            active={chat.id === activeChatId}
            onSelect={() => onSelect(chat.id)}
          />
        ))}
        {!loading && chats.length === 0 && (
          <p className="text-xs text-gray-400 px-2 pt-2">No chats yet</p>
        )}
      </nav>
    </aside>
  );
}
