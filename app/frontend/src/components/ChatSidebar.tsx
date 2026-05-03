import type { Chat } from "../api";

function ChatItem({
  chat,
  active,
  onSelect,
  onDelete,
}: {
  chat: Chat;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={[
        "group flex items-center rounded-md transition-colors",
        active ? "bg-indigo-100" : "hover:bg-gray-100",
      ].join(" ")}
    >
      <button
        onClick={onSelect}
        className={[
          "flex-1 text-left px-3 py-2 text-sm truncate",
          active ? "text-indigo-800 font-medium" : "text-gray-700",
        ].join(" ")}
      >
        {chat.title || "New chat"}
      </button>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="pr-2 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
        title="Delete chat"
      >
        ×
      </button>
    </div>
  );
}

interface Props {
  chats: Chat[];
  loading: boolean;
  activeChatId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
}

export default function ChatSidebar({
  chats,
  loading,
  activeChatId,
  onSelect,
  onNewChat,
  onDelete,
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
            onDelete={() => onDelete(chat.id)}
          />
        ))}
        {!loading && chats.length === 0 && (
          <p className="text-xs text-gray-400 px-2 pt-2">No chats yet</p>
        )}
      </nav>
    </aside>
  );
}
