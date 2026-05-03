interface Props {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
}

export default function ChatInput({ value, disabled, onChange, onSend }: Props) {
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  }

  return (
    <div className="border-t border-gray-200 px-4 py-3 flex gap-2 items-end">
      <textarea
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Ask something about your documents…"
        className="flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50 max-h-40 overflow-y-auto"
        style={{ lineHeight: "1.5" }}
      />
      <button
        onClick={onSend}
        disabled={disabled || !value.trim()}
        className="flex-shrink-0 px-4 py-2 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {disabled ? "…" : "Send"}
      </button>
    </div>
  );
}
