import type { Chat } from "../api";
import styles from "./ChatSidebar.module.css";

interface Props {
  chats: Chat[];
  loading: boolean;
  activeChatId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const min = Math.floor(diff / 60000);
  if (min < 2)  return "just now";
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 24)  return `${hr}h`;
  const d = Math.floor(hr / 24);
  if (d === 1)  return "Yesterday";
  return `${d}d`;
}

export default function ChatSidebar({ chats, loading, activeChatId, onSelect, onNewChat, onDelete }: Props) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.newThreadWrap}>
        <button onClick={onNewChat} className={styles.newThread}>
          <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
          New thread
        </button>
      </div>

      <div className={styles.threadList}>
        {loading && <p className={`mono ${styles.empty}`}>Loading…</p>}
        {chats.map((chat) => {
          const active = chat.id === activeChatId;
          return (
            <div key={chat.id} className={styles.threadItem}>
              <button
                onClick={() => onSelect(chat.id)}
                className={styles.threadBtn}
                data-active={active}
              >
                <div className={styles.threadTitle}>{chat.title || "New thread"}</div>
                <div className={`mono ${styles.threadTime}`}>{timeAgo(chat.updated_at)}</div>
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(chat.id); }}
                title="Delete"
                className={styles.deleteBtn}
              >
                ×
              </button>
            </div>
          );
        })}
        {!loading && chats.length === 0 && (
          <p className={`mono ${styles.empty}`}>No threads yet</p>
        )}
      </div>
    </aside>
  );
}
