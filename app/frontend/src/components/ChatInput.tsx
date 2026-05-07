import styles from "./ChatInput.module.css";

interface Props {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
}

export default function ChatInput({ value, disabled, onChange, onSend }: Props) {
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.key === "Enter" && e.metaKey) || (e.key === "Enter" && !e.shiftKey)) {
      e.preventDefault();
      onSend();
    }
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.inner}>
        <textarea
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Ask a question about a document…"
          className={styles.textarea}
        />
        <button
          onClick={onSend}
          disabled={disabled || !value.trim()}
          className={styles.send}
        >
          {disabled ? "…" : "Send"}
          {!disabled && (
            <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
